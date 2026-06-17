"""Command-line configuration for MGPAN experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


RELATIONS = ["MM", "MG", "GP", "PP", "GM", "PG"]

DEFAULT_METAPATHS = [
    [2, 0],        # M-G-M
    [2, 1, 4, 0],  # M-G-P-G-M
    [3],           # MM
    [1, 4],        # G-P-G
    [0, 2],        # G-M-G
    [4, 1],        # P-G-P
    [4, 0, 2, 1],  # P-G-M-G-P
    [5],           # PP
]


@dataclass
class MGPANConfig:
    dataset: str = "QinN_2014"
    graphdata: str = "QN_graphdataF.bin"
    metadata: str = "QN_graphdata_metaF.pkl"
    subject_ids: str = "subject_ids.csv"
    metapath_cache: str = "QN_metapath_graphsF8"
    experiment: str = "MGPAN_metapath8_typepooling_graphaugment_posweight_cl"
    log: str = "log_MGPAN1"
    model_name: str = "MGPAN"

    seed: int = 66
    n_splits: int = 10
    gnn: str = "sage"
    num_gnn_layer: int = 1
    feat_dim: int = 5
    embed_dim: int = 192
    dim_a: int = 56
    num_node_types: int = 3
    type_emb_dim: int = 4
    node_id_emb_dim: int = 64
    abundance_proj_dim: int = 64

    edge_drop_prob: float = 0.05
    node_drop_prob: float = 0.0
    feat_mask_prob: float = 0.0
    dropout1: float = 0.45
    dropout2: float = 0.45
    dropout3: float = 0.45
    attdropout: float = 0.4
    activation: str = "elu"

    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.004
    accum_steps: int = 1
    num_workers: int = 0
    contrastive_weight: float = 1.0
    min_epochs: int = 70
    patience: int = 10
    min_delta: float = 1e-7

    data_dir: str = "./data"
    metapath_dir: str = "./metapaths"
    saved_model_dir: str = "./saved_models"
    log_dir: str = "./logs"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate MGPAN.")
    parser.add_argument("--dataset", default=MGPANConfig.dataset)
    parser.add_argument("--graphdata", default=MGPANConfig.graphdata)
    parser.add_argument("--metadata", default=MGPANConfig.metadata)
    parser.add_argument("--subject-ids", default=MGPANConfig.subject_ids)
    parser.add_argument("--metapath-cache", default=MGPANConfig.metapath_cache)
    parser.add_argument("--experiment", default=MGPANConfig.experiment)
    parser.add_argument("--log", default=MGPANConfig.log)
    parser.add_argument("--model-name", default=MGPANConfig.model_name)
    parser.add_argument("--seed", type=int, default=MGPANConfig.seed)
    parser.add_argument("--n-splits", type=int, default=MGPANConfig.n_splits)
    parser.add_argument("--gnn", default=MGPANConfig.gnn)
    parser.add_argument("--num-gnn-layer", type=int, default=MGPANConfig.num_gnn_layer)
    parser.add_argument("--embed-dim", type=int, default=MGPANConfig.embed_dim)
    parser.add_argument("--dim-a", type=int, default=MGPANConfig.dim_a)
    parser.add_argument("--edge-drop-prob", type=float, default=MGPANConfig.edge_drop_prob)
    parser.add_argument("--node-drop-prob", type=float, default=MGPANConfig.node_drop_prob)
    parser.add_argument("--feat-mask-prob", type=float, default=MGPANConfig.feat_mask_prob)
    parser.add_argument("--dropout1", type=float, default=MGPANConfig.dropout1)
    parser.add_argument("--dropout2", type=float, default=MGPANConfig.dropout2)
    parser.add_argument("--dropout3", type=float, default=MGPANConfig.dropout3)
    parser.add_argument("--attdropout", type=float, default=MGPANConfig.attdropout)
    parser.add_argument("--activation", default=MGPANConfig.activation)
    parser.add_argument("--batch-size", type=int, default=MGPANConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=MGPANConfig.epochs)
    parser.add_argument("--lr", type=float, default=MGPANConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=MGPANConfig.weight_decay)
    parser.add_argument("--accum-steps", type=int, default=MGPANConfig.accum_steps)
    parser.add_argument("--num-workers", type=int, default=MGPANConfig.num_workers)
    parser.add_argument("--contrastive-weight", type=float, default=MGPANConfig.contrastive_weight)
    parser.add_argument("--min-epochs", type=int, default=MGPANConfig.min_epochs)
    parser.add_argument("--patience", type=int, default=MGPANConfig.patience)
    parser.add_argument("--data-dir", default=MGPANConfig.data_dir)
    parser.add_argument("--metapath-dir", default=MGPANConfig.metapath_dir)
    parser.add_argument("--saved-model-dir", default=MGPANConfig.saved_model_dir)
    parser.add_argument("--log-dir", default=MGPANConfig.log_dir)
    return parser


def parse_args() -> MGPANConfig:
    namespace = build_arg_parser().parse_args()
    valid_names = {field.name for field in fields(MGPANConfig)}
    values = {
        key: value
        for key, value in vars(namespace).items()
        if key in valid_names
    }
    return MGPANConfig(**values)
