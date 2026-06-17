"""Training, evaluation, and prediction routines for MGPAN."""

from __future__ import annotations

import logging
import os
import random
from time import time
from typing import Optional

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loader import set_seed
from utils.metrics import find_best_threshold_with_recall_constraint


class MGPANTrainer:
    """Owns optimization state and evaluation thresholds for one MGPAN fold."""

    def __init__(
        self,
        model: nn.Module,
        device,
        model_dir: str,
        seed: int = 66,
        min_epochs: int = 80,
        patience: int = 10,
        min_delta: float = 1e-7,
        contrastive_weight: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.model_dir = model_dir
        self.seed = seed
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.contrastive_weight = contrastive_weight
        self.best_threshold = None
        self.pos_ratio = None

    def train(
        self,
        train_dataset,
        test_dataset,
        fold: int,
        batch_size: int = 16,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        accum_steps: int = 1,
        num_workers: int = 2,
        pos_class_weight: Optional[float] = 2,
    ):
        set_seed(self.seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.model.to(self.device)

        os.makedirs(self.model_dir, exist_ok=True)
        logging.info("Device: %s", self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=custom_collate,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        if pos_class_weight is not None and pos_class_weight != 1:
            pos_weight = torch.tensor(pos_class_weight, device=self.device, dtype=torch.float)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        contrastive_criterion = SupConLoss(temperature=0.07)
        train_acc, train_p, train_r = [], [], []
        train_f1, train_auc, train_aupr, train_loss = [], [], [], []
        train_score = 0
        best_train_loss = float("inf")
        trigger_times = 0
        start_train = time()

        for epoch in range(epochs):
            self.model.train()
            total_loss_sum = 0.0
            bce_loss_sum = 0.0
            contrastive_loss_sum = 0.0
            n_batches = 0

            data_iter = tqdm(
                train_loader,
                desc=f"Epoch: {epoch:02}",
                total=len(train_loader),
                position=0,
            )

            for batch_idx, (batch_graph, labels, batch_mp_graphs) in enumerate(data_iter):
                batch_graph = batch_graph.to(self.device)
                labels = labels.float().view(-1).to(self.device)

                logits, embed = self.model(batch_graph, mp_graphs_list=batch_mp_graphs)
                if not torch.isfinite(embed).all():
                    raise ValueError("embed contains NaN or Inf")
                if not torch.isfinite(logits).all():
                    print("logits min/max:", logits.min().item(), logits.max().item())
                    raise ValueError("logits contains NaN or Inf")

                if logits.dim() > 1:
                    logits = logits.view(-1)

                bce_loss = criterion(logits, labels)
                contrastive_loss = contrastive_criterion(embed, labels)
                if not torch.isfinite(contrastive_loss):
                    print("NaN/Inf contrastive_loss; skipping batch.")
                    optimizer.zero_grad()
                    continue

                loss_raw = bce_loss + self.contrastive_weight * contrastive_loss
                loss = loss_raw / accum_steps
                if not torch.isfinite(loss):
                    print("NaN/Inf total loss; skipping batch.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                total_loss_sum += loss.item()
                bce_loss_sum += bce_loss.item()
                contrastive_loss_sum += contrastive_loss.item()
                n_batches += 1

                if ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(data_iter)):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if torch.isnan(self.model.abundance_proj[0].weight).any():
                        print("abundance_proj weight has NaN; skipping optimizer.step().")
                        optimizer.zero_grad()
                        continue
                    optimizer.step()
                    optimizer.zero_grad()

                data_iter.set_postfix(
                    {
                        "train_score": train_score,
                        "avg_loss": total_loss_sum / max(n_batches, 1),
                    }
                )

            if n_batches == 0:
                raise RuntimeError("No valid training batches were processed.")

            avg_total_loss = total_loss_sum / n_batches
            avg_bce_loss = bce_loss_sum / n_batches
            avg_contrastive_loss = contrastive_loss_sum / n_batches

            acc, precision, recall, f1, auc_value, aupr = self.evaluate(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                flag=False,
            )
            train_score = acc
            logging.info(
                "[Fold %s] Epoch %02d Summary: Avg Total Loss: %.4f, "
                "Avg BCE Loss: %.4f, Avg Contrastive Loss: %.4f",
                fold + 1,
                epoch + 1,
                avg_total_loss,
                avg_bce_loss,
                avg_contrastive_loss,
            )
            logging.info(
                "[Fold %s] Epoch %02d: Acc: %.4f | Prec: %.4f | "
                "Recall: %.4f | F1: %.4f | AUC: %.4f | AUPR: %.4f",
                fold + 1,
                epoch + 1,
                acc,
                precision,
                recall,
                f1,
                auc_value,
                aupr,
            )

            test_metrics = self.evaluate(
                test_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                flag=True,
            )
            logging.info(
                "[Fold %s] Test Epoch %02d: Acc: %.4f | Prec: %.4f | "
                "Recall: %.4f | F1: %.4f | AUC: %.4f | AUPR: %.4f",
                fold + 1,
                epoch + 1,
                test_metrics[0],
                test_metrics[1],
                test_metrics[2],
                test_metrics[3],
                test_metrics[4],
                test_metrics[10],
            )

            scheduler.step()
            train_loss.append(avg_total_loss)
            train_acc.append(acc)
            train_p.append(precision)
            train_r.append(recall)
            train_f1.append(f1)
            train_auc.append(auc_value)
            train_aupr.append(aupr)

            torch.save(
                {
                    "epoch": epoch,
                    "loss": avg_total_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.model_dir, "checkpoint.pt"),
            )

            if best_train_loss - avg_total_loss > self.min_delta:
                best_train_loss = avg_total_loss
                trigger_times = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_train_loss,
                    },
                    os.path.join(self.model_dir, f"best_model_fold{fold}.pt"),
                )
                logging.info(
                    "[EarlyStopping] New best train_loss: %.4f at epoch %s",
                    best_train_loss,
                    epoch,
                )
            elif epoch >= self.min_epochs:
                trigger_times += 1
                logging.info(
                    "[EarlyStopping] No improvement for %s/%s epochs",
                    trigger_times,
                    self.patience,
                )
                if trigger_times >= self.patience:
                    logging.info("[EarlyStopping] Triggered at epoch %s", epoch)
                    checkpoint = torch.load(
                        os.path.join(self.model_dir, f"best_model_fold{fold}.pt"),
                        map_location=self.device,
                    )
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    break

        logging.info("Total training time: %.2fs", time() - start_train)
        model_name = os.path.normpath(self.model_dir).split(os.sep)[-1]
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f"{model_name}.pt"))

        return train_acc, train_p, train_r, train_f1, train_auc, train_loss, train_aupr

    def evaluate(
        self,
        eval_dataset,
        batch_size: int = 16,
        num_workers: int = 2,
        flag: bool = False,
    ):
        self.model.eval()
        self.model.to(self.device)

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=custom_collate,
        )

        pred_probs, y_true = self.predict(eval_loader)
        pred_probs = np.asarray(pred_probs)
        y_true = np.asarray(y_true)

        if np.isnan(pred_probs).any() or np.isinf(pred_probs).any():
            raise ValueError("pred_probs contains NaN or Inf")

        if flag is False:
            best_thresh, _, _, _ = find_best_threshold_with_recall_constraint(y_true, pred_probs)
            self.best_threshold = best_thresh
            self.pos_ratio = (pred_probs >= best_thresh).mean()
            y_pred = (pred_probs >= best_thresh).astype(int)
        else:
            if self.pos_ratio is None:
                raise ValueError("Run train evaluation before test evaluation.")
            adaptive_thresh = np.quantile(pred_probs, 1 - self.pos_ratio)
            print(f"[Test] adaptive threshold={adaptive_thresh:.4f}")
            y_pred = (pred_probs >= adaptive_thresh).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_value = roc_auc_score(y_true, pred_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, pred_probs)
        aupr = auc(recall_curve, precision_curve)

        if flag is False:
            return accuracy, precision, recall, f1, auc_value, aupr

        fpr, tpr, _ = roc_curve(y_true, pred_probs)
        cm = confusion_matrix(y_true, y_pred)
        return (
            accuracy,
            precision,
            recall,
            f1,
            auc_value,
            fpr,
            tpr,
            precision_curve,
            recall_curve,
            cm,
            aupr,
        )

    def predict(self, graph_loader):
        self.model.eval()
        self.model.to(self.device)
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_graph, batch_labels, batch_mp_graphs in tqdm(
                graph_loader,
                desc="Predicting",
                total=len(graph_loader),
            ):
                batch_graph = batch_graph.to(self.device)
                logits, _ = self.model(batch_graph, mp_graphs_list=batch_mp_graphs)
                batch_preds = torch.sigmoid(logits)
                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(batch_labels.view(-1).tolist())

        return all_preds, all_labels

    def extract_raw_predictions(self, test_dataset, fold, batch_size, num_workers):
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=custom_collate,
        )

        pred_probs, y_true = self.predict(test_loader)
        return pd.DataFrame(
            {
                "Fold": fold + 1,
                "y_true": np.asarray(y_true).astype(int),
                "y_prob": np.asarray(pred_probs).astype(float),
            }
        )


class SupConLoss(nn.Module):
    """Supervised contrastive loss used by the original training objective."""

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.size(0)
        if batch_size == 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = 1.0 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        positive_per_sample = mask.sum(1)
        if (positive_per_sample == 0).all():
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits_max, _ = torch.max(logits * logits_mask, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (positive_per_sample + self.eps)
        return -mean_log_prob_pos.mean()


def custom_collate(batch):
    graphs, labels, mp_graphs_list = zip(*batch)
    batched_graph = dgl.batch(graphs)

    if torch.is_tensor(labels[0]):
        labels = torch.stack([label.detach().clone() for label in labels])
    else:
        labels = torch.tensor(labels)

    return batched_graph, labels, list(mp_graphs_list)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
