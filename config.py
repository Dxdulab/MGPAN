"""Central configuration for MGPAN experiments."""

import argparse
from dataclasses import dataclass


RELATIONS = ["MM", "MG", "GP", "PP", "GM", "PG"]

DEFAULT_METAPATHS = [
    [2, 0],        # M-G-M
    [2, 1, 4, 0],  # M-G-P-G-M
    [3],           # MM
    [1, 4],        # G-P-G
    [0, 2],        # G-M-G
    [4, 1],        # P-G-P
    [4, 0, 2, 1],  # P-G-M-G-P
    [5]            # PP
]

  


@dataclass
class MGPANConfig:
    # Dataset and output paths
    dataset: str = "QinN_2014"
    graphdata: str = "QN_graphdataF.bin"
    metadata: str = "QN_graphdata_metaF.pkl"
    subject_ids: str = "subject_ids.csv"
    data_dir: str = "./data"
    metapath_dir: str = "./metapaths"
    saved_model_dir: str = "saved_models"
    log_dir: str = "logs"

    # Experiment naming
    metapath: str = "F_metapath_graphs8(full)"
    experimental: str = "Fmetapath8+typepooling+graphaugment+pos_weight(4+64+64)+0.1cl"
    log: str = "log_Mp_MGPAN"
    model_name: str = "MGPAN"

    # Reproducibility and cross validation
    seed: int = 66
    device: str = "auto"
    n_splits: int = 10
    fold_shuffle: bool = True
    run_last_folds_first: int = 0
    graph_build_device: str = "cpu"
    mean_curve_points: int = 100

    # Graph construction and model input
    feat_dim: int = 5
    abundance_input_dim: int = 3
    num_node_types: int = 3
    type_emb_dim: int = 8
    type_emb_hidden_dim: int = 16
    node_id_emb_dim: int = 64
    abundance_proj_dim: int = 64

    # GNN and attention model
    gnn: str = "sage"
    num_gnn_layer: int = 1
    embed_dim: int = 192
    dim_a: int = 64
    dropout1: float = 0.35
    dropout2: float = 0.35
    dropout3: float = 0.35
    attdropout: float = 0.35
    activation: str = "gelu"
    classifier_dropout: float = 0.35
    graph_pool_hidden_dim: int = 32
    graph_readout_num_types: int = 3
    gat_num_heads: int = 1
    sage_aggregator: str = "mean"
    residual_dropout: float = 0.2

    # Graph augmentation
    edge_drop_prob: float = 0.05
    node_drop_prob: float = 0.0
    feat_mask_prob: float = 0.0
    bidirected: bool = False

    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0003
    accum_steps: int = 1
    num_workers: int = 4
    pos_class_weight: float = 1.0
    auto_pos_class_weight: bool = True
    contrastive_weight: float = 0.5
    contrastive_temperature: float = 0.07
    contrastive_eps: float = 1e-8
    scheduler_eta_min: float = 1e-6
    grad_clip_norm: float = 1.0
    min_epochs: int = 80
    patience: int = 10
    min_delta: float = 1e-7
    train_loader_shuffle: bool = True
    train_pin_memory: bool = False
    eval_pin_memory: bool = True

    # Evaluation thresholding
    threshold_min_recall: float = 0.6
    threshold_eps: float = 1e-12


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.casefold()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def add_bool_argument(parser, name, default, help_text):
    parser.add_argument(
        name,
        nargs="?",
        const=True,
        default=default,
        type=str_to_bool,
        help=help_text,
    )


def build_parser(config=None):
    config = config or MGPANConfig()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=config.dataset, help="Name of the dataset.")
    parser.add_argument("--graphdata", type=str, default=config.graphdata, help="Name of the graph data file.")
    parser.add_argument("--metadata", type=str, default=config.metadata, help="Name of the metadata file.")
    parser.add_argument("--subject-ids", type=str, default=config.subject_ids, help="Name of the subject id file.")
    parser.add_argument("--data-dir", type=str, default=config.data_dir, help="Root directory for input data.")
    parser.add_argument("--metapath-dir", type=str, default=config.metapath_dir, help="Root directory for metapath caches.")
    parser.add_argument("--saved-model-dir", type=str, default=config.saved_model_dir, help="Root directory for saved models.")
    parser.add_argument("--log-dir", type=str, default=config.log_dir, help="Root directory for logs.")

    parser.add_argument("--metapath", type=str, default=config.metapath, help="Name of the metapath cache folder.")
    parser.add_argument("--experimental", type=str, default=config.experimental, help="Experiment label used in outputs.")
    parser.add_argument("--log", type=str, default=config.log, help="Log file name without extension.")
    parser.add_argument("--model-name", "--model_name", dest="model_name", type=str, default=config.model_name, help="Name to save the trained model.")

    parser.add_argument("--seed", type=int, default=config.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default=config.device, help='Training device: "auto", "cpu", or "cuda".')
    parser.add_argument("--n-splits", type=int, default=config.n_splits, help="Number of CV folds.")
    add_bool_argument(parser, "--fold-shuffle", config.fold_shuffle, "Shuffle CV folds.")
    parser.add_argument("--run-last-folds-first", type=int, default=config.run_last_folds_first, help="Run the last N folds before the earlier folds.")
    parser.add_argument("--graph-build-device", type=str, default=config.graph_build_device, help="Device used while rebuilding node ids.")
    parser.add_argument("--mean-curve-points", type=int, default=config.mean_curve_points, help="Number of points in mean ROC/PR curves.")

    parser.add_argument("--feat-dim", type=int, default=config.feat_dim, help="Input feature dimension.")
    parser.add_argument("--abundance-input-dim", type=int, default=config.abundance_input_dim, help="Number of abundance feature columns.")
    parser.add_argument("--num-node-types", type=int, default=config.num_node_types, help="Number of node types.")
    parser.add_argument("--type-emb-dim", type=int, default=config.type_emb_dim, help="Node-type embedding dimension.")
    parser.add_argument("--type-emb-hidden-dim", type=int, default=config.type_emb_hidden_dim, help="Hidden dimension for type embedding MLP.")
    parser.add_argument("--node-id-emb-dim", type=int, default=config.node_id_emb_dim, help="Node id embedding dimension.")
    parser.add_argument("--abundance-proj-dim", type=int, default=config.abundance_proj_dim, help="Abundance projection dimension.")

    parser.add_argument("--gnn", type=str, default=config.gnn, help='GNN layer type: "sage", "gatv2", or "gin".')
    parser.add_argument("--num-gnn-layer", type=int, default=config.num_gnn_layer, help="Number of GNN layers.")
    parser.add_argument("--embed-dim", type=int, default=config.embed_dim, help="Output embedding dimension.")
    parser.add_argument("--dim-a", type=int, default=config.dim_a, help="Attention hidden dimension.")
    parser.add_argument("--dropout1", type=float, default=config.dropout1, help="Dropout rate before graph embedding.")
    parser.add_argument("--dropout2", type=float, default=config.dropout2, help="Dropout rate for pooling.")
    parser.add_argument("--dropout3", type=float, default=config.dropout3, help="Dropout rate in GNN layers.")
    parser.add_argument("--attdropout", type=float, default=config.attdropout, help="Dropout rate in attention.")
    parser.add_argument("--activation", type=str, default=config.activation, help='Activation: "relu", "elu", or "gelu".')
    parser.add_argument("--classifier-dropout", type=float, default=config.classifier_dropout, help="Classifier dropout.")
    parser.add_argument("--graph-pool-hidden-dim", type=int, default=config.graph_pool_hidden_dim, help="Node-type pooling hidden dimension.")
    parser.add_argument("--graph-readout-num-types", type=int, default=config.graph_readout_num_types, help="Number of node-type readout blocks.")
    parser.add_argument("--gat-num-heads", type=int, default=config.gat_num_heads, help="Number of GATv2 heads.")
    parser.add_argument("--sage-aggregator", type=str, default=config.sage_aggregator, help="SAGE aggregator type.")
    parser.add_argument("--residual-dropout", type=float, default=config.residual_dropout, help="Residual dropout in MGPAN layers.")

    parser.add_argument("--edge-drop-prob", "--edge_drop_prob", dest="edge_drop_prob", type=float, default=config.edge_drop_prob, help="Edge dropout probability.")
    parser.add_argument("--node-drop-prob", "--node_drop_prob", dest="node_drop_prob", type=float, default=config.node_drop_prob, help="Node dropout probability.")
    parser.add_argument("--feat-mask-prob", "--feat_mask_prob", dest="feat_mask_prob", type=float, default=config.feat_mask_prob, help="Feature masking probability.")
    add_bool_argument(parser, "--bidirected", config.bidirected, "Use a bidirectional version of input graphs.")

    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Maximum training epochs.")
    parser.add_argument("--lr", type=float, default=config.lr, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay, help="L2 regularization penalty.")
    parser.add_argument("--accum-steps", type=int, default=config.accum_steps, help="Gradient accumulation steps.")
    parser.add_argument("--num-workers", type=int, default=config.num_workers, help="Number of DataLoader workers.")
    parser.add_argument("--pos-class-weight", type=float, default=config.pos_class_weight, help="Positive class loss weight.")
    add_bool_argument(parser, "--auto-pos-class-weight", config.auto_pos_class_weight, "Compute positive class weight from each training fold.")
    parser.add_argument("--contrastive-weight", type=float, default=config.contrastive_weight, help="Contrastive loss weight.")
    parser.add_argument("--contrastive-temperature", type=float, default=config.contrastive_temperature, help="SupCon temperature.")
    parser.add_argument("--contrastive-eps", type=float, default=config.contrastive_eps, help="SupCon numerical epsilon.")
    parser.add_argument("--scheduler-eta-min", type=float, default=config.scheduler_eta_min, help="Minimum LR for cosine scheduler.")
    parser.add_argument("--grad-clip-norm", type=float, default=config.grad_clip_norm, help="Gradient clipping norm.")
    parser.add_argument("--min-epochs", type=int, default=config.min_epochs, help="Minimum epochs before early stopping.")
    parser.add_argument("--patience", type=int, default=config.patience, help="Early stopping patience.")
    parser.add_argument("--min-delta", type=float, default=config.min_delta, help="Minimum improvement for early stopping.")
    add_bool_argument(parser, "--train-loader-shuffle", config.train_loader_shuffle, "Shuffle training DataLoader.")
    add_bool_argument(parser, "--train-pin-memory", config.train_pin_memory, "Pin memory for training DataLoader.")
    add_bool_argument(parser, "--eval-pin-memory", config.eval_pin_memory, "Pin memory for evaluation DataLoader.")

    parser.add_argument("--threshold-min-recall", type=float, default=config.threshold_min_recall, help="Minimum recall when selecting train threshold.")
    parser.add_argument("--threshold-eps", type=float, default=config.threshold_eps, help="Numerical epsilon for threshold F1.")

    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
