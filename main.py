"""Main entry point for MGPAN cross-validation experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from config import DEFAULT_METAPATHS, RELATIONS, MGPANConfig, parse_args
from model import MGPAN, MGPANTrainer
from utils.data_loader import (
    GraphDataset,
    GraphMetapathDataset,
    assign_processed_features,
    augment_dataset,
    build_id_maps_and_features_with_rank_optimized,
    build_or_load_metapath_cache,
    load_graphs_and_metadata,
    load_subject_ids,
    make_fold_data,
    set_seed,
    to_homogeneous_graphs,
)
from utils.metrics import (
    plot_cross_validation_summary,
    plot_test_evaluation,
    plot_training_curves,
    summarize_cv_curves,
)


def main(config: Optional[MGPANConfig] = None) -> None:
    if config is None:
        config = parse_args()
    run_cross_validation(config)


def run_cross_validation(config: MGPANConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = configure_logging(config)

    raw_graphs, homogeneous_graphs, labels, meta = prepare_base_graphs(config)
    subject_ids = load_subject_ids(config.data_dir, config.dataset, config.subject_ids)
    if len(subject_ids) != len(raw_graphs):
        raise ValueError(
            f"Subject count {len(subject_ids)} does not match graph count {len(raw_graphs)}."
        )

    labels_np = labels.cpu().numpy().reshape(-1)
    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(
        splitter.split(
            X=np.zeros(len(labels_np)),
            y=labels_np,
            groups=subject_ids,
        )
    )
    run_order = list(range(len(folds) - 2, len(folds))) + list(range(0, len(folds) - 2))

    output_model_dir = Path(config.saved_model_dir) / config.dataset / "metapaths" / config.log
    output_model_dir.mkdir(parents=True, exist_ok=True)
    figure_base_dir = Path(config.data_dir) / config.dataset / f"figures_{config.log}"
    metapath_cache_dir = Path(config.metapath_dir) / config.dataset / config.metapath_cache

    fold_results = []
    all_fpr, all_tpr, all_prec, all_recall, all_cm, all_auc = [], [], [], [], [], []
    raw_prediction_frames = []

    for fold in run_order:
        train_idx, test_idx = folds[fold]
        logging.info(config)

        fold_data = make_fold_data(
            homogeneous_graphs,
            labels,
            meta,
            train_idx,
            test_idx,
            device="cpu",
        )
        log_label_distribution("Train", fold_data["train_labels"])
        log_label_distribution("Test", fold_data["test_labels"])

        train_dataset = GraphDataset(
            "train_dataset",
            fold_data["train_graphs"],
            fold_data["train_labels"],
        )
        test_dataset = GraphDataset(
            "test_dataset",
            fold_data["test_graphs"],
            fold_data["test_labels"],
        )
        train_dataset_aug = augment_dataset(
            train_dataset,
            edge_drop_prob=config.edge_drop_prob,
            node_drop_prob=config.node_drop_prob,
            feat_mask_prob=config.feat_mask_prob,
        )

        train_dataset_aug, train_mp_graphs, test_dataset, test_mp_graphs = (
            build_or_load_metapath_cache(
                train_dataset=train_dataset_aug,
                test_dataset=test_dataset,
                metapaths=DEFAULT_METAPATHS,
                cache_dir=metapath_cache_dir,
                fold=fold,
            )
        )
        train_dataset_mp = GraphMetapathDataset(
            name="train_dataset_mp",
            graphs=train_dataset_aug.graphs,
            labels=train_dataset_aug.labels,
            mp_graphs_list=train_mp_graphs,
        )
        test_dataset_mp = GraphMetapathDataset(
            name="test_dataset_mp",
            graphs=test_dataset.graphs,
            labels=test_dataset.labels,
            mp_graphs_list=test_mp_graphs,
        )

        model = build_model(config, num_node_ids=fold_data["num_node_ids"])
        num_pos = torch.sum(fold_data["train_labels"] == 1).item()
        num_neg = torch.sum(fold_data["train_labels"] == 0).item()
        pos_class_weight = num_neg / max(num_pos, 1)
        logging.info("pos_class_weight: %s", pos_class_weight)

        trainer = MGPANTrainer(
            model=model,
            device=device,
            model_dir=str(output_model_dir),
            seed=config.seed,
            min_epochs=config.min_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
            contrastive_weight=config.contrastive_weight,
        )

        train_acc, train_p, train_r, train_f1, train_auc, train_loss, train_aupr = trainer.train(
            train_dataset=train_dataset_mp,
            test_dataset=test_dataset_mp,
            fold=fold,
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            accum_steps=config.accum_steps,
            num_workers=config.num_workers,
            pos_class_weight=pos_class_weight,
        )

        write_fold_metrics(
            log_file,
            fold,
            "Validation",
            {
                "loss": train_loss,
                "Accuracies": train_acc,
                "Precisions": train_p,
                "Recalls": train_r,
                "F1s": train_f1,
                "AUCs": train_auc,
                "AUPRs": train_aupr,
            },
        )
        fold_figure_dir = figure_base_dir / f"fold_{fold + 1}"
        plot_training_curves(train_loss, train_auc, save_dir=fold_figure_dir)

        test_metrics = trainer.evaluate(
            test_dataset_mp,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            flag=True,
        )
        (
            test_acc,
            test_p,
            test_r,
            test_f1,
            test_auc,
            fpr,
            tpr,
            precision_curve,
            recall_curve,
            cm,
            test_aupr,
        ) = test_metrics
        raw_prediction_frames.append(
            trainer.extract_raw_predictions(
                test_dataset=test_dataset_mp,
                fold=fold,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
        )

        write_fold_metrics(
            log_file,
            fold,
            "Test",
            {
                "Accuracy": f"{test_acc:.4f}",
                "Precision": f"{test_p:.4f}",
                "Recall": f"{test_r:.4f}",
                "F1": f"{test_f1:.4f}",
                "AUC": f"{test_auc:.4f}",
                "AUPR": f"{test_aupr:.4f}",
                "Confusion Matrix": f"\n{cm}",
            },
        )

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_prec.append(precision_curve)
        all_recall.append(recall_curve)
        all_cm.append(cm)
        all_auc.append(test_auc)
        plot_test_evaluation(
            fpr,
            tpr,
            precision_curve,
            recall_curve,
            cm,
            test_auc,
            save_dir=fold_figure_dir,
        )

        fold_results.append(
            {
                "train_acc": train_acc[-1],
                "train_precision": train_p[-1],
                "train_recall": train_r[-1],
                "train_f1": train_f1[-1],
                "train_auc": train_auc[-1],
                "test_acc": test_acc,
                "test_precision": test_p,
                "test_recall": test_r,
                "test_f1": test_f1,
                "test_auc": test_auc,
                "test_aupr": test_aupr,
            }
        )

    figure_base_dir.mkdir(parents=True, exist_ok=True)
    final_predictions = pd.concat(raw_prediction_frames, ignore_index=True)
    prediction_path = figure_base_dir / f"{config.experiment}_raw_predictions.csv"
    final_predictions.to_csv(prediction_path, index=False)
    print(f"Raw prediction data saved to: {prediction_path}")

    logging.info("===== %s-Fold CV Summary =====", config.n_splits)
    logging.info(config)
    logging.info("experiment: %s", config.experiment)

    for metric in [
        "train_acc",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_auc",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "test_aupr",
    ]:
        values = [result[metric] for result in fold_results]
        logging.info("%s per fold: %s", metric, [f"{value:.4f}" for value in values])
        logging.info("%s: %.4f +/- %.4f", metric, np.mean(values), np.std(values))

    curve_summary = summarize_cv_curves(
        all_fpr=all_fpr,
        all_tpr=all_tpr,
        all_prec=all_prec,
        all_recall=all_recall,
        all_cm=all_cm,
        all_auc=all_auc,
    )
    plot_cross_validation_summary(
        curve_summary["mean_fpr"],
        curve_summary["mean_tpr"],
        curve_summary["std_tpr"],
        curve_summary["mean_prec"],
        curve_summary["std_prec"],
        curve_summary["mean_recall"],
        curve_summary["mean_cm"],
        curve_summary["mean_auc"],
        curve_summary["std_auc"],
        save_dir=figure_base_dir / "mean_summary",
    )


def prepare_base_graphs(config: MGPANConfig):
    all_graphs, all_labels, all_meta = load_graphs_and_metadata(
        data_dir=config.data_dir,
        dataset=config.dataset,
        graphdata=config.graphdata,
        metadata=config.metadata,
    )
    homogeneous_graphs = to_homogeneous_graphs(all_graphs)
    processed_features = build_id_maps_and_features_with_rank_optimized(homogeneous_graphs)
    assign_processed_features(homogeneous_graphs, processed_features)
    return all_graphs, homogeneous_graphs, all_labels, all_meta


def build_model(config: MGPANConfig, num_node_ids: int) -> MGPAN:
    return MGPAN(
        gnn_type=config.gnn,
        num_gnn_layers=config.num_gnn_layer,
        relations=RELATIONS,
        feat_dim=config.feat_dim,
        embed_dim=config.embed_dim,
        dim_a=config.dim_a,
        dropout1=config.dropout1,
        dropout2=config.dropout2,
        dropout3=config.dropout3,
        attdropout=config.attdropout,
        activation=config.activation,
        num_node_types=config.num_node_types,
        type_emb_dim=config.type_emb_dim,
        num_node_ids=num_node_ids,
        node_id_emb_dim=config.node_id_emb_dim,
        abundance_proj_dim=config.abundance_proj_dim,
        metapaths=DEFAULT_METAPATHS,
    )


def configure_logging(config: MGPANConfig) -> Path:
    log_path = Path(config.log_dir) / config.dataset / "metapaths"
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{config.log}.out"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=str(log_file),
        filemode="a",
        force=True,
    )
    logging.info(config)
    return log_file


def log_label_distribution(prefix: str, labels) -> None:
    label_values = [int(label.item()) if torch.is_tensor(label) else int(label) for label in labels]
    unique_values, counts = np.unique(label_values, return_counts=True)
    logging.info("%s sample count: %s", prefix, len(label_values))
    for value, count in zip(unique_values, counts):
        logging.info("  Label %s: %s (%.2f%%)", value, count, 100 * count / len(label_values))


def write_fold_metrics(log_file: Path, fold: int, title: str, values: dict) -> None:
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("-" * 25 + "\n")
        handle.write(f"Fold {fold + 1} {title} metrics:\n")
        for key, value in values.items():
            handle.write(f"{key}: {value}\n")
        handle.write("-" * 25 + "\n\n")


if __name__ == "__main__":
    main()
