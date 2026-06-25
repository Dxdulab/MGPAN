"""Core MGPAN model and training helpers."""

import logging
import os
from collections import defaultdict
from time import time
import pandas as pd
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.attention import GATv2ConvEdgeOnly, MetaPathAttention, NodeTypeAwarePooling
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GINConv, SAGEConv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.optim.lr_scheduler import StepLR

from config import MGPANConfig
from utils.data_loader import set_seed


_DEFAULT_CONFIG = MGPANConfig()

class MGPAN(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout1,
            dropout2,
            dropout3,
            attdropout,
            activation,
            num_node_types,
            type_emb_dim,
            num_node_ids,
            node_id_emb_dim,
            abundance_proj_dim,
            metapaths,
            type_emb_hidden_dim=_DEFAULT_CONFIG.type_emb_hidden_dim,
            abundance_input_dim=_DEFAULT_CONFIG.abundance_input_dim,
            classifier_dropout=_DEFAULT_CONFIG.classifier_dropout,
            graph_pool_hidden_dim=_DEFAULT_CONFIG.graph_pool_hidden_dim,
            graph_readout_num_types=_DEFAULT_CONFIG.graph_readout_num_types,
            gat_num_heads=_DEFAULT_CONFIG.gat_num_heads,
            sage_aggregator=_DEFAULT_CONFIG.sage_aggregator,
            residual_dropout=_DEFAULT_CONFIG.residual_dropout


    ):
        super(MGPAN, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(relations)
        self.metapaths = metapaths
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.num_node_types=num_node_types
        self.num_node_ids=num_node_ids
        self.abundance_input_dim = abundance_input_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.attdropout=attdropout
        self.activation = activation.casefold()
        self.embedder = MGPANGraph(
            gnn_type=self.gnn_type,
            num_gnn_layers=self.num_gnn_layers,
            relations=self.relations,
            feat_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            dim_a=self.dim_a,
            dropout2=self.dropout2,
            dropout3=self.dropout3,
            attdropout=self.attdropout,
            activation=self.activation,
            metapaths=self.metapaths,
            graph_pool_hidden_dim=graph_pool_hidden_dim,
            graph_readout_num_types=graph_readout_num_types,
            gat_num_heads=gat_num_heads,
            sage_aggregator=sage_aggregator,
            residual_dropout=residual_dropout
        )
        self.type_emb = nn.Embedding(num_node_types,type_emb_dim)
        self.type_emb_mlp = nn.Sequential(
            nn.Linear(type_emb_dim, type_emb_hidden_dim),
            nn.ReLU(),
            nn.Linear(type_emb_hidden_dim, type_emb_dim)
        )
        self.node_id_emb = nn.Embedding(num_node_ids+1, node_id_emb_dim)
        self.node_id_emb_mlp = nn.Sequential(
            nn.Linear(node_id_emb_dim, node_id_emb_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout1)
        )
        self.abundance_proj = nn.Sequential(
            nn.Linear(abundance_input_dim, abundance_proj_dim),
            nn.ReLU(),
            nn.Linear(abundance_proj_dim, abundance_proj_dim),
            nn.ReLU()
        )

        total_input_dim = abundance_proj_dim +type_emb_dim+node_id_emb_dim
        self.node_feat_proj = nn.Linear(total_input_dim, self.embed_dim)
        self.node_feat_norm = nn.LayerNorm(self.embed_dim)
        
        final_embed_dim=int(self.embed_dim)
        self.classifier = MinimalClassifier(embed_dim=final_embed_dim, dropout=classifier_dropout)
    def forward(self, graph,mp_graphs_list=None, return_attn=False):
        feat = graph.ndata['feat'].float()
        x = feat[:, :self.abundance_input_dim]
        abundance_feats = self.abundance_proj(x)
        node_type=feat[:, self.abundance_input_dim].long() 
        type_feats = self.type_emb_mlp(self.type_emb(node_type))
        node_ids=feat[:, self.abundance_input_dim + 1].long()
        unk_id = self.num_node_ids
        node_ids = torch.where(node_ids >= unk_id, torch.tensor(unk_id, device=node_ids.device), node_ids)

        node_id_feats = self.node_id_emb(node_ids)
        node_id_feats  = self.node_id_emb_mlp(node_id_feats)

        h = torch.cat([abundance_feats,type_feats,node_id_feats], dim=1)

        h = self.node_feat_proj(h)

        h = self.node_feat_norm(h)
   
        h = F.relu(h)

        h = F.dropout(h, p=self.dropout1, training=self.training)

        if return_attn:
            embed, att_dict = self.embedder(
                graph, h,
                return_attn=True,
                mp_graphs_list=mp_graphs_list
            )
            out = self.classifier(embed)
            return out, embed, att_dict
        else:
            embed = self.embedder(graph, h, mp_graphs_list=mp_graphs_list)
            out = self.classifier(embed)
            return out, embed


    def train_model(
            self,
            train_dataset,
            test_dataset,
            fold,
            batch_size=_DEFAULT_CONFIG.batch_size,
            EPOCHS=_DEFAULT_CONFIG.epochs,
            lr=_DEFAULT_CONFIG.lr,
            weight_decay=_DEFAULT_CONFIG.weight_decay,
            accum_steps=_DEFAULT_CONFIG.accum_steps,
            num_workers=_DEFAULT_CONFIG.num_workers,
            pos_class_weight=_DEFAULT_CONFIG.pos_class_weight,
            device='cpu',
            model_dir=None,
            model_name=_DEFAULT_CONFIG.model_name,
            seed=_DEFAULT_CONFIG.seed,
            contrastive_weight=_DEFAULT_CONFIG.contrastive_weight,
            contrastive_temperature=_DEFAULT_CONFIG.contrastive_temperature,
            contrastive_eps=_DEFAULT_CONFIG.contrastive_eps,
            scheduler_eta_min=_DEFAULT_CONFIG.scheduler_eta_min,
            grad_clip_norm=_DEFAULT_CONFIG.grad_clip_norm,
            min_epochs=_DEFAULT_CONFIG.min_epochs,
            patience=_DEFAULT_CONFIG.patience,
            min_delta=_DEFAULT_CONFIG.min_delta,
            threshold_min_recall=_DEFAULT_CONFIG.threshold_min_recall,
            threshold_eps=_DEFAULT_CONFIG.threshold_eps,
            train_loader_shuffle=_DEFAULT_CONFIG.train_loader_shuffle,
            train_pin_memory=_DEFAULT_CONFIG.train_pin_memory,
            eval_pin_memory=_DEFAULT_CONFIG.eval_pin_memory
    ):
        if model_dir is None:
            model_dir = os.path.join(_DEFAULT_CONFIG.saved_model_dir, _DEFAULT_CONFIG.model_name)

        set_seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        self.to(device)

        os.makedirs(model_dir, exist_ok=True)
        logging.info(f'Device: {device}')
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=scheduler_eta_min
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_loader_shuffle,
            num_workers=num_workers,
            pin_memory=train_pin_memory,
            collate_fn=custom_collate,
            worker_init_fn=seed_worker,   # ⭐ 确保 DataLoader 多线程可复现
            generator=g                   # ⭐ 确保 shuffle 可复现
        )

        start_train = time()
        train_acc, train_p, train_r, train_f1,train_auc,train_aupr,train_loss = [], [], [], [],[],[],[]
        train_score = 0
        if pos_class_weight is not None and pos_class_weight != 1:
            pos_weight_tensor = torch.tensor(pos_class_weight, device=device, dtype=torch.float)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        contrastive_criterion = SupConLoss(
            temperature=contrastive_temperature,
            eps=contrastive_eps
        )
        best_train_loss = float('inf')
        trigger_times = 0
        
        for epoch in range(EPOCHS):
            self.train()
            self.to(device)

            data_iter = tqdm(
                train_loader,
                desc=f'Epoch: {epoch:02}',
                total=len(train_loader),
                position=0
            )

            raw_loss_sum, bce_loss_sum, focal_loss_sum,contrastive_loss_sum,ranking_loss_sum = 0.0, 0.0, 0.0 ,0.0,0.0# accumulate raw (un-divided) loss for reporting
            n_batches = 0
            for i, (batch_graph, labels,batch_mp_graphs) in enumerate(data_iter):
                batch_graph = batch_graph.to(device)
                labels = labels.float().view(-1).to(device)  # ensure 1D float

                logits,embed= self(batch_graph,mp_graphs_list=batch_mp_graphs)


                if torch.isnan(embed).any() or torch.isinf(embed).any():
                    raise ValueError("embed contains NaN or Inf")

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("logits min/max:", logits.min().item(), logits.max().item())
                    raise ValueError("logits contains NaN or Inf")

               
                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.view(-1)
                elif logits.dim() > 1 and logits.size(0) > 1 and logits.size(1) != 1:
                    print("WARNING: logits shape unexpected:", logits.shape)
                    logits = logits.view(-1)
                if torch.isnan(logits).any():
                    raise ValueError("logits contains NaN")

                bce_loss = criterion(logits, labels)
             
                contrastive_loss = contrastive_criterion(embed, labels)
         
               
                loss_raw = bce_loss + contrastive_weight*contrastive_loss


                if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                    print("❌ NaN/Inf contrastive_loss — skip batch")
                    optimizer.zero_grad()
                    continue

                loss = loss_raw / accum_steps
                loss.backward()
                if torch.isnan(loss) or torch.isinf(loss):
                    print("❌ NaN/Inf total loss — skip batch")
                    optimizer.zero_grad()
                    continue

                raw_loss_sum += loss.item()
                bce_loss_sum += bce_loss.item()
                contrastive_loss_sum += contrastive_loss.item()
                n_batches += 1


                if ((i + 1) % accum_steps == 0) or ((i + 1) == len(data_iter)):

                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        max_norm=grad_clip_norm
                    )

                    if torch.isnan(self.abundance_proj[0].weight).any():
                        print("❌ abundance_proj Linear weight NaN — skip optimizer.step()")
                        optimizer.zero_grad()
                        continue

                    optimizer.step()
                    optimizer.zero_grad()

                data_iter.set_postfix({
                    'train_score': train_score,
                    'avg_loss': raw_loss_sum / n_batches
                })
            
            avg_total_loss = raw_loss_sum / n_batches
            avg_bce_loss = bce_loss_sum / n_batches
            avg_contrastive_loss = contrastive_loss_sum / n_batches

            acc, p, r, f1,auc,aupr = self.eval_model(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                flag=False,
                min_recall=threshold_min_recall,
                threshold_eps=threshold_eps,
                pin_memory=eval_pin_memory
            )
            train_score = acc
            logging.info(
                f"[Fold {fold+1}] Epoch {epoch+1:02} Summary: "
                f"Avg Total Loss: {avg_total_loss:.4f}, "
                f"Avg BCE Loss: {avg_bce_loss  :.4f}, "
                f"Avg Contrastive Loss: {avg_contrastive_loss:.4f} "
            )
   

            logging.info(f'[Fold {fold+1}]: Epoch{epoch+1:02}:  Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}| AUPR: {aupr:.4f}')
            
            test_acc, test_p, test_r, test_f1,test_auc,_,_,_,_,_,test_aupr= self.eval_model(
                    test_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    device=device,
                    flag=True,
                    min_recall=threshold_min_recall,
                    threshold_eps=threshold_eps,
                    pin_memory=eval_pin_memory
            )
            logging.info(f'[Fold {fold+1}]: test Epoch{epoch+1:02}: Acc: {test_acc:.4f} | Prec: {test_p:.4f} | Recall: {test_r:.4f} | F1: {test_f1:.4f}| AUC: {test_auc:.4f}| AUPR: {test_aupr:.4f}')
            scheduler.step()
            train_loss.append(avg_total_loss)
            train_acc.append(acc)
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f1)
            train_auc.append(auc)
            train_aupr.append(aupr)
            torch.save({
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, f'{model_dir}/checkpoint.pt')
                

            if best_train_loss - avg_total_loss > min_delta:
                best_train_loss = avg_total_loss
                trigger_times = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': best_train_loss
                }, f'{model_dir}/best_model_fold{fold}.pt')
                logging.info(f"[EarlyStopping] 🟢 New best train_loss: {best_train_loss:.4f} at epoch {epoch}")
            else:
                if epoch >= min_epochs:  # 🔹加入最低训练轮数限制
                    trigger_times += 1
                    logging.info(f"[EarlyStopping] No improvement for {trigger_times}/{patience} epochs")
                    if trigger_times >= patience:
                        logging.info(f"[EarlyStopping] 🛑 Triggered at epoch {epoch}")
                        checkpoint = torch.load(f'{model_dir}/best_model_fold{fold}.pt', map_location=device)
                        self.load_state_dict(checkpoint['model_state_dict'])
                        break
        
        end_train = time()
        logging.info(f'Total training time... {end_train - start_train:.2f}s')

        torch.save(self.state_dict(), f'{model_dir}/{model_name}.pt')

        return train_acc, train_p, train_r, train_f1,train_auc,train_loss,train_aupr
    



    def find_best_threshold_with_recall_constraint(
        self,
        y_true,
        pred_probs,
        min_recall=_DEFAULT_CONFIG.threshold_min_recall,
        eps=_DEFAULT_CONFIG.threshold_eps
    ):
        precision, recall, thresholds = precision_recall_curve(y_true, pred_probs)

        precision = precision[:-1]
        recall = recall[:-1]

        f1 = 2 * precision * recall / (precision + recall + eps)

        valid = recall >= min_recall

        if valid.sum() == 0:
            best_idx = np.argmax(f1)
        else:
            best_idx = np.argmax(f1 * valid)

        best_threshold = thresholds[best_idx]

        return best_threshold, f1[best_idx], precision[best_idx], recall[best_idx]
    
    def eval_model(
        self,
        eval_dataset,
        batch_size=_DEFAULT_CONFIG.batch_size,
        num_workers=_DEFAULT_CONFIG.num_workers,
        device='cuda',
        flag=False,   # False=train, True=test
        min_recall=_DEFAULT_CONFIG.threshold_min_recall,
        threshold_eps=_DEFAULT_CONFIG.threshold_eps,
        pin_memory=_DEFAULT_CONFIG.eval_pin_memory
    ):
        """
        flag=False (train):
            - Recall约束 + F1最优阈值
            - 保存 threshold + 正类比例
            - 返回: acc, prec, rec, f1, auc

        flag=True (test):
            - 使用 Quantile 自适应阈值（分布漂移）
            - 返回: acc, prec, rec, f1, auc, fpr, tpr, precision_curve, recall_curve, cm
        """

        self.eval()
        self.to(device)

        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=custom_collate
        )

        pred_probs, y_true = self.predict(eval_loader, device=device)

        pred_probs = np.asarray(pred_probs)
        y_true = np.asarray(y_true)

        if np.isnan(pred_probs).any():
            raise ValueError("pred_probs contains NaN")
        if np.isinf(pred_probs).any():
            raise ValueError("pred_probs contains Inf")

        if flag is False:
            best_thresh, best_f1,_,_ = \
                self.find_best_threshold_with_recall_constraint(
                    y_true,
                    pred_probs,     # ⭐ 可以调
                    min_recall=min_recall,
                    eps=threshold_eps
                )

            self.best_threshold = best_thresh

            self.pos_ratio = (pred_probs >= best_thresh).mean()


            y_pred = (pred_probs >=  best_thresh).astype(int)

        else:
            if not hasattr(self, "pos_ratio"):
                raise ValueError("Run train eval first!")

            adaptive_thresh = np.quantile(pred_probs, 1 - self.pos_ratio)

            print(f"[Test] adaptive threshold={adaptive_thresh:.4f}")

            y_pred = (pred_probs >= adaptive_thresh).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc1 = roc_auc_score(y_true, pred_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, pred_probs)
        aupr = auc(recall_curve, precision_curve)
        if flag is False:
            return accuracy, precision, recall, f1, auc1,aupr
        else:
            fpr, tpr, _ = roc_curve(y_true, pred_probs)
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, pred_probs
            )
            cm = confusion_matrix(y_true, y_pred)
            return (
                accuracy,
                precision,
                recall,
                f1,
                auc1,
                fpr,
                tpr,
                precision_curve,
                recall_curve,
                cm,
                aupr
            )



    def predict(self, graph_loader, device='cpu'):
        self.eval()
        self.to(device)

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_graph, batch_labels, batch_mp_graphs in tqdm(graph_loader, desc='Predicting', total=len(graph_loader)):
                batch_graph = batch_graph.to(device)
                
                logits, _ = self(batch_graph, mp_graphs_list=batch_mp_graphs)
                batch_preds = torch.sigmoid(logits)  # 只对 logits 做 sigmoid
                all_preds.extend(batch_preds.cpu().tolist())
                all_labels.extend(batch_labels.tolist())

        return all_preds, all_labels


class MGPANGraph(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout2=0,
            dropout3=0,
            attdropout=0,
            activation=None,
            metapaths=None,
            graph_pool_hidden_dim=_DEFAULT_CONFIG.graph_pool_hidden_dim,
            graph_readout_num_types=_DEFAULT_CONFIG.graph_readout_num_types,
            gat_num_heads=_DEFAULT_CONFIG.gat_num_heads,
            sage_aggregator=_DEFAULT_CONFIG.sage_aggregator,
            residual_dropout=_DEFAULT_CONFIG.residual_dropout
    ):
        super(MGPANGraph, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.activation = activation
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.attdropout = attdropout
        self.metapaths = metapaths
        self.layers = nn.ModuleList([
            MGPANLayer(
                gnn_type=self.gnn_type,
                relations=self.relations,
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dim_a=self.dim_a,
                attdropout=self.attdropout,
                dropout3=self.dropout3,
                activation=self.activation,
                metapaths=self.metapaths,
                gat_num_heads=gat_num_heads,
                sage_aggregator=sage_aggregator,
                residual_dropout=residual_dropout
            )
        ])
        for _ in range(1, self.num_gnn_layers):
            self.layers.append(
                MGPANLayer(
                    gnn_type=self.gnn_type,
                    relations=self.relations,
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    dim_a=self.dim_a,
                    attdropout=self.attdropout,
                    dropout3=self.dropout3,
                    activation=self.activation,
                    metapaths=self.metapaths,
                    gat_num_heads=gat_num_heads,
                    sage_aggregator=sage_aggregator,
                    residual_dropout=residual_dropout
                )
            )
        self.type_att_pool = NodeTypeAwarePooling(
            embed_dim=embed_dim,
            att_hidden_dim=graph_pool_hidden_dim,
            att_dropout=self.attdropout
        )
        self.graph_feat_proj = nn.Linear(embed_dim * graph_readout_num_types, int(embed_dim))

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn
    
    def forward(self, graph, feat,node_type=None, return_attn=False,mp_graphs_list=None):

        
        h = feat
        for i, layer in enumerate(self.layers):
            h = layer(graph, h,mp_graphs_list=mp_graphs_list)
        

        if return_attn:
            h_Tyattn, att_dict = self.type_att_pool(graph, h, return_attn=True)
        else:
            h_Tyattn = self.type_att_pool(graph, h)
        h_readout = h_Tyattn
        h_readout = F.relu(self.graph_feat_proj(h_readout))

        if return_attn:
            return h_readout, att_dict
        else:
            return h_readout        




class MGPANLayer(nn.Module):
    def __init__(self,
                 gnn_type,
                 relations,
                 in_dim,
                 out_dim,
                 dim_a,
                 attdropout,
                 dropout3=0.0,
                 activation='relu',
                 metapaths=None,
                 gat_num_heads=_DEFAULT_CONFIG.gat_num_heads,
                 sage_aggregator=_DEFAULT_CONFIG.sage_aggregator
                 ):
        super(MGPANLayer, self).__init__()
        self.gnn_type = gnn_type
        self.relations = relations  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout3=dropout3
        self.attdropout=attdropout
        self.activation = self._get_activation_fn(activation)
        self.num_metapaths=len(metapaths)

        self.mp_layers = nn.ModuleList()
        for _ in range(self.num_metapaths):
            if self.gnn_type == 'gin':
                conv = GINConv(
                    apply_func=nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.Dropout(p=self.dropout3),
                        self.activation,
                        nn.Linear(out_dim, out_dim),
                        nn.Dropout(p=self.dropout3),
                        self.activation,
                    ),
                    aggregator_type='sum',
                )
            elif self.gnn_type == 'gatv2':
                conv = GATv2ConvEdgeOnly(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    num_heads=gat_num_heads,           # 如果需要多头，可以改为 >1
                    feat_drop=self.dropout3,
                    attn_drop=self.dropout3,
                    residual=False,
                    activation=self.activation,
                    allow_zero_in_degree=True
                )
            elif self.gnn_type == 'sage':
                conv = SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    aggregator_type=sage_aggregator,  # 可选 'mean', 'pool', 'lstm', 'gcn'
                    feat_drop=self.dropout3,
                    activation=self.activation
                )
            else:
                raise ValueError(f"Invalid gnn_type: {gnn_type}. Choose 'gin' or 'sage'.")
            self.mp_layers.append(conv)

        self.attention = MetaPathAttention(
            num_metapaths=self.num_metapaths,
            embed_dim=out_dim,   # out_dim 就是 GNN 层输出的维度
            dim_a=self.dim_a,            # 可调，hidden_dim
            dropout=self.attdropout          # 可调
        )

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            return None
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f'Invalid activation function: {activation}')


    def forward(self, graph, feat, mp_graphs_list, device=None):

        if device is None:
            device = graph.device

        mp_node_embs = []

        num_metapaths = len(self.mp_layers)

        for i, gnn_layer in enumerate(self.mp_layers):

            graphs_i = []
            for sample_graphs in mp_graphs_list:
                gi = sample_graphs[i].to(device)

                if gi.num_nodes() > 0 :
                    graphs_i.append(gi)

            if len(graphs_i) == 0:
                continue

            batched_mp_graph = dgl.batch(graphs_i)

            assert feat.shape[0] == batched_mp_graph.num_nodes(), \
                f"[Metapath {i}] feat mismatch: {feat.shape[0]} vs {batched_mp_graph.num_nodes()}"

            feat_mp = torch.nn.functional.dropout(
                feat,
                p=self.dropout3,                   # 推荐 0.2~0.3
                training=self.training
            )

            if isinstance(gnn_layer, GATv2ConvEdgeOnly):
                h_out = gnn_layer(
                    batched_mp_graph,
                    feat_mp,
                    edge_weight=batched_mp_graph.edata.get("weight", None)
                )
                if h_out.dim() == 3:
                    h_out = h_out.mean(dim=1)
            else:
                h_out = gnn_layer(
                    batched_mp_graph,
                    feat_mp,
                    edge_weight=batched_mp_graph.edata.get("weight", None)
                )

            mp_node_embs.append(h_out.unsqueeze(0))

        if len(mp_node_embs) == 0:
            return torch.zeros(
                feat.shape[0],
                self.out_dim,
                device=device
            )

        h_views = torch.cat(mp_node_embs, dim=0)

        h_views = torch.nn.functional.layer_norm(
            h_views,
            h_views.shape[-1:]
        )

        h_views = torch.nn.functional.dropout(
            h_views,
            p=self.dropout3,                       # 推荐 0.3~0.4
            training=self.training
        )

        fused_h = self.attention(h_views)

        h_out=fused_h + feat

        return h_out


class MinimalClassifier(nn.Module):
    def __init__(self, embed_dim, dropout=_DEFAULT_CONFIG.classifier_dropout):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1) 
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


def custom_collate(batch, *args, **kwargs):
    graphs, labels, mp_graphs_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels, list(mp_graphs_list)

def seed_worker(worker_id):
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SupConLoss(nn.Module):
    def __init__(
        self,
        temperature: float = _DEFAULT_CONFIG.contrastive_temperature,
        eps: float = _DEFAULT_CONFIG.contrastive_eps
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        device = features.device
        features = F.normalize(features, dim=1)               # (B, D) -> L2-normalize each row
        batch_size = features.size(0)
        if batch_size == 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)  # (B, B)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)   # (B, B), 1 where same class
        logits_mask = 1.0 - torch.eye(batch_size, device=device)  # mask out self-similarity
        mask = mask * logits_mask

        positive_per_sample = mask.sum(1)
        if (positive_per_sample == 0).all():
            return torch.tensor(0.0, device=device, requires_grad=True)

        logits_max, _ = torch.max(logits * logits_mask, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (positive_per_sample + self.eps)
        loss = - mean_log_prob_pos.mean()
        return loss



def check_tensor(x, name):
    if not torch.isfinite(x).all():
        print(f"[NaN DETECTED] {name}")
        print("min:", x.min().item(), "max:", x.max().item())
        raise ValueError(f"{name} contains NaN or Inf")

