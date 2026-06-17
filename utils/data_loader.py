"""Data loading, preprocessing, augmentation, and meta-path graph construction."""

from __future__ import annotations

import copy
import pickle
import random
from pathlib import Path
from typing import Iterable, Sequence, Union

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset


class GraphDataset(DGLDataset):
    """DGL dataset wrapper for graph-level labels."""

    def __init__(self, name: str, graphs: Sequence[dgl.DGLGraph], labels):
        super().__init__(name=name)
        self.graphs = list(graphs)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass


class GraphMetapathDataset(DGLDataset):
    """DGL dataset wrapper that returns the base graph and its meta-path graphs."""

    def __init__(self, name: str, graphs, labels, mp_graphs_list=None):
        super().__init__(name=name)
        self.graphs = list(graphs)
        self.labels = labels
        self.mp_graphs_list = mp_graphs_list

        if self.mp_graphs_list is not None and len(self.mp_graphs_list) != len(self.graphs):
            raise ValueError("mp_graphs_list and graphs must have the same length.")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        mp_graphs = None
        if self.mp_graphs_list is not None:
            mp_graphs = self.mp_graphs_list[idx]
        return self.graphs[idx], self.labels[idx], mp_graphs

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass


def set_seed(seed: int = 66) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        print(f"[NaN DETECTED] {name}")
        print("min:", x.min().item(), "max:", x.max().item())
        raise ValueError(f"{name} contains NaN or Inf")


def load_graphs_and_metadata(data_dir: str, dataset: str, graphdata: str, metadata: str):
    dataset_dir = Path(data_dir) / dataset
    graphs, label_dict = dgl.load_graphs(str(dataset_dir / graphdata))
    labels = label_dict["labels"]

    with (dataset_dir / metadata).open("rb") as handle:
        meta = pickle.load(handle)

    return graphs, labels, meta


def load_subject_ids(data_dir: str, dataset: str, subject_ids_file: str) -> list:
    path = Path(data_dir) / dataset / subject_ids_file
    return pd.read_csv(path)["subject_id"].tolist()


def to_homogeneous_graphs(graphs: Iterable[dgl.DGLGraph]) -> list[dgl.DGLGraph]:
    homogeneous_graphs = []

    for idx, graph in enumerate(graphs):
        graph_homo = dgl.to_homogeneous(
            graph,
            ndata=["feat"],
            edata=["weight"],
            store_type=True,
        )
        check_tensor(graph_homo.ndata["feat"], f"raw feat graph {idx}")
        homogeneous_graphs.append(graph_homo)

    return homogeneous_graphs


def build_id_maps_and_features_with_rank_optimized(
    all_graphs,
    feat_key: str = "feat",
    type_key: str = "_TYPE",
):
    eps = 1e-6
    feats_proc = []

    for graph in all_graphs:
        feat = graph.ndata[feat_key].float()
        node_type = graph.ndata[type_key]

        scaled_feat = torch.zeros_like(feat)
        types_in_graph = torch.unique(node_type)
        for node_t in types_in_graph:
            mask = node_type == node_t
            scaled_feat[mask] = feat[mask]

        feat = scaled_feat
        log_feat = torch.log(feat + eps)
        feat_norm = torch.zeros_like(feat)
        rank_feat = torch.zeros_like(feat)
        log_centered = torch.zeros_like(log_feat)

        for node_t in types_in_graph:
            mask = node_type == node_t
            feat_t = feat[mask]
            feat_norm[mask] = (feat_t - feat_t.mean()) / (feat_t.std() + eps)

            ranks = torch.argsort(torch.argsort(feat_t.view(-1)))
            rank_feat[mask] = ranks.float().unsqueeze(1) / (len(feat_t) - 1 + eps)

            log_feat_t = log_feat[mask]
            log_centered[mask] = log_feat_t - log_feat_t.mean()

        type_scalar = node_type.float().unsqueeze(1)
        feat_final = torch.cat([feat_norm, log_centered, rank_feat, type_scalar], dim=1)
        feats_proc.append(feat_final)

    return feats_proc


def assign_processed_features(graphs: Sequence[dgl.DGLGraph], features: Sequence[torch.Tensor]) -> None:
    for idx, (graph, feat) in enumerate(zip(graphs, features)):
        check_tensor(feat, f"processed feat graph {idx}")
        graph.ndata["feat"] = feat


def graph_augment(
    graph: dgl.DGLGraph,
    feat_key: str = "feat",
    edge_drop_prob: float = 0.1,
    node_drop_prob: float = 0.05,
    feat_mask_prob: float = 0.05,
) -> dgl.DGLGraph:
    num_nodes = graph.num_nodes()
    node_mask = torch.rand(num_nodes, device=graph.device) > node_drop_prob
    if not node_mask.any():
        node_mask[0] = True

    nodes_kept = node_mask.nonzero(as_tuple=False).squeeze()
    graph_aug = dgl.node_subgraph(graph, nodes_kept)

    num_edges = graph_aug.num_edges()
    if num_edges > 0:
        edge_mask = torch.rand(num_edges, device=graph.device) > edge_drop_prob
        edges_kept = edge_mask.nonzero(as_tuple=False).squeeze()
        graph_aug = dgl.edge_subgraph(graph_aug, edges_kept, preserve_nodes=True)

    if feat_key in graph_aug.ndata:
        feat = graph_aug.ndata[feat_key]
        mask = torch.rand(feat.shape, device=graph.device) < feat_mask_prob
        feat_aug = feat.clone()
        feat_aug[mask] = 0.0
        graph_aug.ndata[feat_key] = feat_aug

    return graph_aug


def augment_dataset(
    dataset: GraphDataset,
    edge_drop_prob: float = 0.1,
    node_drop_prob: float = 0.05,
    feat_mask_prob: float = 0.05,
) -> GraphDataset:
    graphs_aug = [
        graph_augment(
            graph,
            feat_key="feat",
            edge_drop_prob=edge_drop_prob,
            node_drop_prob=node_drop_prob,
            feat_mask_prob=feat_mask_prob,
        )
        for graph in dataset.graphs
    ]
    return GraphDataset("train_dataset_aug", graphs_aug, dataset.labels)


def build_etype_adjacencies_from_homo(
    graph: dgl.DGLGraph,
    num_etypes: int,
    weight_key: str = "weight",
) -> list[sp.csr_matrix]:
    num_nodes = graph.num_nodes()
    src, dst = graph.edges(order="eid")
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    if dgl.ETYPE not in graph.edata:
        raise RuntimeError("graph.edata must contain dgl.ETYPE.")

    etypes = graph.edata[dgl.ETYPE].cpu().numpy()
    if weight_key in graph.edata:
        weights = graph.edata[weight_key].cpu().numpy().astype(np.float32)
    else:
        weights = np.ones(len(src), dtype=np.float32)

    rows = [[] for _ in range(num_etypes)]
    cols = [[] for _ in range(num_etypes)]
    vals = [[] for _ in range(num_etypes)]

    for source, target, etype, weight in zip(src, dst, etypes, weights):
        if 0 <= etype < num_etypes:
            rows[int(etype)].append(int(source))
            cols[int(etype)].append(int(target))
            vals[int(etype)].append(float(weight))

    adjs = []
    for etype in range(num_etypes):
        adj = sp.coo_matrix(
            (vals[etype], (rows[etype], cols[etype])),
            shape=(num_nodes, num_nodes),
            dtype=np.float32,
        ).tocsr()
        adjs.append(adj)

    return adjs


def metapath_to_coo(adjs: Sequence[sp.csr_matrix], metapath: Sequence[int]):
    if not metapath:
        raise ValueError("metapath cannot be empty.")

    adjacency = adjs[metapath[0]].copy()
    for etype in metapath[1:]:
        adjacency = adjacency.dot(adjs[etype])

    adjacency = adjacency.tocoo()
    if adjacency.nnz == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    return (
        adjacency.row.astype(np.int64),
        adjacency.col.astype(np.int64),
        adjacency.data.astype(np.float32),
    )


def build_metapath_graphs(
    graph: dgl.DGLGraph,
    metapaths: Sequence[Sequence[int]],
    num_etypes: int = 6,
    weight_key: str = "weight",
    device=None,
    add_self_loop_if_empty: bool = False,
) -> list[dgl.DGLGraph]:
    num_nodes = graph.num_nodes()
    adjs = build_etype_adjacencies_from_homo(
        graph,
        num_etypes=num_etypes,
        weight_key=weight_key,
    )

    graph_kwargs = {"num_nodes": num_nodes}
    if device is not None:
        graph_kwargs["device"] = device

    mp_graphs = []
    for metapath in metapaths:
        src, dst, weights = metapath_to_coo(adjs, metapath)
        graph_mp = dgl.graph((src, dst), **graph_kwargs)

        id_tensor = torch.arange(num_nodes, dtype=torch.long)
        weight_tensor = torch.from_numpy(weights).float()
        if device is not None:
            id_tensor = id_tensor.to(device)
            weight_tensor = weight_tensor.to(device)

        graph_mp.ndata["_ID"] = id_tensor
        graph_mp.edata["weight"] = weight_tensor

        if add_self_loop_if_empty and graph_mp.num_edges() == 0:
            graph_mp = dgl.add_self_loop(graph_mp)
            graph_mp.edata["weight"] = torch.ones(
                graph_mp.num_edges(),
                dtype=torch.float32,
                device=graph_mp.device,
            )

        mp_graphs.append(graph_mp)

    return mp_graphs


def build_or_load_metapath_cache(
    train_dataset: GraphDataset,
    test_dataset: GraphDataset,
    metapaths: Sequence[Sequence[int]],
    cache_dir: Union[str, Path],
    fold: int,
) -> tuple[GraphDataset, list, GraphDataset, list]:
    cache_dir = Path(cache_dir)
    train_path = cache_dir / f"fold_{fold}_train.pkl"
    test_path = cache_dir / f"fold_{fold}_test.pkl"

    if train_path.exists() and test_path.exists():
        with train_path.open("rb") as handle:
            train_dataset_cached, train_mp_graphs = pickle.load(handle)
        with test_path.open("rb") as handle:
            test_dataset_cached, test_mp_graphs = pickle.load(handle)
        return train_dataset_cached, train_mp_graphs, test_dataset_cached, test_mp_graphs

    train_mp_graphs = [
        build_metapath_graphs(graph, metapaths)
        for graph in train_dataset.graphs
    ]
    test_mp_graphs = [
        build_metapath_graphs(graph, metapaths, device="cpu")
        for graph in test_dataset.graphs
    ]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as handle:
        pickle.dump((train_dataset, train_mp_graphs), handle)
    with test_path.open("wb") as handle:
        pickle.dump((test_dataset, test_mp_graphs), handle)

    return train_dataset, train_mp_graphs, test_dataset, test_mp_graphs


def build_node_id_global(
    train_graphs,
    test_graphs,
    train_meta,
    test_meta,
    device: str = "cpu",
):
    feat_dim = train_graphs[0].ndata["feat"].shape[1]
    node2id_global = {}
    global_id_counter = 0

    for meta in train_meta:
        for ntype in ["gene", "microbe", "pathway"]:
            for name in meta[ntype]:
                if name not in node2id_global:
                    node2id_global[name] = global_id_counter
                    global_id_counter += 1

    def add_id_as_feature(graphs, meta_list, is_train, start_counter=None):
        nonlocal node2id_global, global_id_counter

        for graph, meta in zip(graphs, meta_list):
            ntype_tensor = graph.ndata[dgl.NTYPE]
            nid_tensor = graph.ndata[dgl.NID]
            num_nodes = graph.num_nodes()
            orig_feat = graph.ndata["feat"]

            if orig_feat.shape[1] != feat_dim:
                pad = torch.zeros(
                    (num_nodes, feat_dim - orig_feat.shape[1]),
                    dtype=orig_feat.dtype,
                    device=device,
                )
                orig_feat = torch.cat([orig_feat, pad], dim=1)

            id_feat = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)
            temp_counter = start_counter if start_counter is not None else global_id_counter

            for node_t in [0, 1, 2]:
                mask = ntype_tensor == node_t
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue

                local_ids = nid_tensor[idx].tolist()
                if node_t == 0:
                    names = [meta["gene"][node_id] for node_id in local_ids]
                elif node_t == 1:
                    names = [meta["microbe"][node_id] for node_id in local_ids]
                else:
                    names = [meta["pathway"][node_id] for node_id in local_ids]

                ids = []
                for name in names:
                    if is_train:
                        if name not in node2id_global:
                            raise RuntimeError(f"New training node was not indexed: {name}")
                        ids.append(node2id_global[name])
                    elif name in node2id_global:
                        ids.append(node2id_global[name])
                    else:
                        ids.append(temp_counter)
                        temp_counter += 1

                id_feat[idx] = torch.tensor(ids, dtype=torch.float32, device=device).unsqueeze(1)

            graph.ndata["feat"] = torch.cat([orig_feat, id_feat], dim=1)

    add_id_as_feature(train_graphs, train_meta, is_train=True)

    for graph, meta in zip(test_graphs, test_meta):
        add_id_as_feature([graph], [meta], is_train=False, start_counter=global_id_counter)

    return node2id_global, train_graphs, test_graphs


def make_fold_data(
    all_graphs,
    all_labels,
    all_meta,
    train_idx,
    test_idx,
    device: str = "cpu",
):
    train_graphs = [copy.deepcopy(all_graphs[i]) for i in train_idx]
    test_graphs = [copy.deepcopy(all_graphs[i]) for i in test_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_labels = [all_labels[i] for i in test_idx]
    train_meta = [all_meta[i] for i in train_idx]
    test_meta = [all_meta[i] for i in test_idx]

    train_labels_tensor = torch.stack(train_labels)
    test_labels_tensor = torch.stack(test_labels)

    node2id_global, train_graphs, test_graphs = build_node_id_global(
        train_graphs,
        test_graphs,
        train_meta,
        test_meta,
        device=device,
    )

    for idx, graph in enumerate(train_graphs):
        check_tensor(graph.ndata["feat"], f"train graph {idx} feat")
    for idx, graph in enumerate(test_graphs):
        check_tensor(graph.ndata["feat"], f"test graph {idx} feat")

    return {
        "train_graphs": train_graphs,
        "test_graphs": test_graphs,
        "train_labels": train_labels_tensor,
        "test_labels": test_labels_tensor,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "num_node_ids": len(node2id_global),
    }
