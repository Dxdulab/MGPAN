"""Data loading, graph augmentation, and meta-path construction utilities."""

import copy
import os
import random
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from dgl.data import DGLDataset
class GraphmpDatasetMP(DGLDataset):
    def __init__(
        self,
        name,
        graphs,
        labels,
        mp_graphs_list=None,
        microbe_counts=None,
        pathway_counts=None,          # 👈 新增
    ):
        """
        graphs: list[DGLGraph]
        labels: Tensor or list, shape (N,)
        mp_graphs_list: list[list[DGLGraph]] or None
        microbe_counts: Tensor or list, shape (N,)
        pathway_counts: Tensor or list, shape (N, 2) -> [MG, PG]
        """
        super(GraphmpDatasetMP, self).__init__(name=name)

        self.graphs = graphs
        self.labels = labels
        self.mp_graphs_list = mp_graphs_list

        if microbe_counts is None:
            raise ValueError("microbe_counts must be provided")

        if not torch.is_tensor(microbe_counts):
            microbe_counts = torch.tensor(
                microbe_counts, dtype=torch.float32
            )

        if pathway_counts is None:
            raise ValueError("pathway_counts must be provided")

        if not torch.is_tensor(pathway_counts):
            pathway_counts = torch.tensor(
                pathway_counts, dtype=torch.float32
            )

        self.microbe_counts = microbe_counts
        self.pathway_counts = pathway_counts

        assert len(self.graphs) == len(self.labels) \
               == len(self.microbe_counts) == len(self.pathway_counts), \
            "graphs / labels / microbe_counts / pathway_counts 长度不一致"

        if self.mp_graphs_list is not None:
            assert len(self.mp_graphs_list) == len(self.graphs), \
                "mp_graphs_list 与 graphs 长度不一致"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        返回五元组：
        graph, label, mp_graphs, microbe_count, edge_count
        """
        g = self.graphs[idx]
        y = self.labels[idx]

        mp_graphs = None
        if self.mp_graphs_list is not None:
            mp_graphs = self.mp_graphs_list[idx]

        microbe_count = self.microbe_counts[idx]
        pathway_count = self.pathway_counts[idx]   # shape (2,)

        return g, y, mp_graphs, microbe_count, pathway_count

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass


class GraphmpDataset(DGLDataset):
    def __init__(self, name, graphs, labels, mp_graphs_list=None):
        """
        graphs: list of DGLGraph
        labels: list or tensor of labels
        mp_graphs_list: list of list of metapath graphs, 与 graphs 一一对应
        """
        super(GraphmpDataset, self).__init__(name=name)
        self.graphs = graphs
        self.labels = labels
        self.mp_graphs_list = mp_graphs_list  # 新增，用于存放每个图的 metapath 子图

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        返回三元组：graph, label, mp_graphs_list
        """
        g = self.graphs[idx]
        y = self.labels[idx]
        mp_graphs = None
        if self.mp_graphs_list is not None:
            mp_graphs = self.mp_graphs_list[idx]
        return g, y, mp_graphs

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass

class GraphDataset(DGLDataset):
    def __init__(self, name, graphs, labels):
        super(GraphDataset, self).__init__(name=name)
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels
        
    def process(self):
        pass


    
def graph_augment(g, feat_key='feat', edge_drop_prob=0.1, node_drop_prob=0.05, feat_mask_prob=0.05):
    """
    Args:
        g: DGLGraph
        feat_key: g.ndata中存储特征的键名 (默认为 'feat')
    """
    num_nodes = g.num_nodes()
    
    node_mask = torch.rand(num_nodes, device=g.device) > node_drop_prob
    if not node_mask.any(): # 如果mask全为False，强制保留至少一个节点或不进行丢弃
        node_mask[0] = True 
        
    nodes_kept = node_mask.nonzero(as_tuple=False).squeeze()
    
    g_aug = dgl.node_subgraph(g, nodes_kept)

    num_edges = g_aug.num_edges()
    if num_edges > 0: # 只有当有边的时候才处理
        edge_mask = torch.rand(num_edges, device=g.device) > edge_drop_prob
        edges_kept = edge_mask.nonzero(as_tuple=False).squeeze()
        
        g_aug = dgl.edge_subgraph(g_aug, edges_kept, preserve_nodes=True)

    if feat_key in g_aug.ndata:
        feat = g_aug.ndata[feat_key]
        mask = torch.rand(feat.shape, device=g.device) < feat_mask_prob
        
        feat_aug = feat.clone()
        feat_aug[mask] = 0.0
        g_aug.ndata[feat_key] = feat_aug
    
    return g_aug


def augment_dataset(dataset, edge_drop_prob=0.1, node_drop_prob=0.05, feat_mask_prob=0.05):
    graphs_aug = []
    labels_aug = []

    for g, label in zip(dataset.graphs, dataset.labels):
        feat = g.ndata['feat']
        g_aug= graph_augment(g, feat_key='feat',
                                        edge_drop_prob=edge_drop_prob,
                                        node_drop_prob=node_drop_prob,
                                        feat_mask_prob=feat_mask_prob)
        graphs_aug.append(g_aug)
        labels_aug.append(label)
    
    return GraphDataset('my_dataset_aug', graphs_aug, labels_aug)

def build_etype_adjacencies_from_homo(
    graph,
    num_etypes: int,
    weight_key: str = "weight"
):
    """
    返回长度为 num_etypes 的 CSR adjacency matrix 列表
    每个 adjacency 的 shape = (N, N)
    """
    N = graph.num_nodes()

    src, dst = graph.edges(order="eid")
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    if dgl.ETYPE not in graph.edata:
        raise RuntimeError("graph.edata 必须包含 dgl.ETYPE")

    etypes = graph.edata[dgl.ETYPE].cpu().numpy()

    if weight_key in graph.edata:
        weights = graph.edata[weight_key].cpu().numpy().astype(np.float32)
    else:
        weights = np.ones(len(src), dtype=np.float32)

    adjs = [sp.csr_matrix((N, N), dtype=np.float32) for _ in range(num_etypes)]

    for s, t, e, w in zip(src, dst, etypes, weights):
        if 0 <= e < num_etypes:
            adjs[e][int(s), int(t)] += w

    return adjs


def metapath_to_coo(
    adjs,
    metapath,
    num_nodes: int
):
    """
    metapath: list[int]
    return: src, dst, weight (numpy)
    """
    if len(metapath) == 0:
        raise ValueError("metapath 不能为空")

    A = adjs[metapath[0]].copy()
    for et in metapath[1:]:
        A = A.dot(adjs[et])

    A = A.tocoo()

    if A.nnz == 0:
        src = np.empty((0,), dtype=np.int64)
        dst = np.empty((0,), dtype=np.int64)
        w = np.empty((0,), dtype=np.float32)
    else:
        src = A.row.astype(np.int64)
        dst = A.col.astype(np.int64)
        w = A.data.astype(np.float32)

    return src, dst, w


def build_metapath_graphs(
    graph,
    metapaths,
    num_etypes: int = 6,
    weight_key: str = "weight",
    device=None,
    add_self_loop_if_empty: bool = False
):
    """
    返回：List[DGLGraph]
    """
    N = graph.num_nodes()
    adjs = build_etype_adjacencies_from_homo(
        graph,
        num_etypes=num_etypes,
        weight_key=weight_key
    )

    mp_graphs = []

    for mp in metapaths:
        src, dst, w = metapath_to_coo(adjs, mp, num_nodes=N)

        g_mp = dgl.graph(
            (src, dst),
            num_nodes=N,
            device=device
        )
        g_mp.ndata["_ID"] = torch.arange(
            N,
            dtype=torch.long,
            device=device
        )
        g_mp.edata["weight"] = torch.from_numpy(w).to(device)

        if add_self_loop_if_empty and g_mp.num_edges() == 0:
            g_mp = dgl.add_self_loop(g_mp)
            g_mp.edata["weight"] = torch.ones(
                g_mp.num_edges(),
                dtype=torch.float32,
                device=device
            )

        mp_graphs.append(g_mp)

    return mp_graphs



def build_node_id_global(train_graphs, test_graphs,train_meta, test_meta,device="cpu"):
    """
    CV 版本（固定特征维度）：
    - 训练集构建全局唯一 ID
    - 测试集每张图单独扩展 ID，从训练集最大 ID + 1 开始
    - 原始节点特征维度保持不变，只新增最后一列 ID
    """

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

        for g, meta in zip(graphs, meta_list):
            ntype_tensor = g.ndata[dgl.NTYPE]
            nid_tensor = g.ndata[dgl.NID]
            num_nodes = g.num_nodes()

            orig_feat = g.ndata["feat"]
            if orig_feat.shape[1] != feat_dim:
                pad = torch.zeros((num_nodes, feat_dim - orig_feat.shape[1]),
                                  dtype=orig_feat.dtype, device=device)
                orig_feat = torch.cat([orig_feat, pad], dim=1)

            id_feat = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

            temp_counter = start_counter if start_counter is not None else global_id_counter

            for t in [0, 1, 2]:
                mask = (ntype_tensor == t)
                idx = mask.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue

                local_ids = nid_tensor[idx].tolist()
                if t == 0:
                    names = [meta["gene"][i] for i in local_ids]
                elif t == 1:
                    names = [meta["microbe"][i] for i in local_ids]
                else:
                    names = [meta["pathway"][i] for i in local_ids]

                ids = []
                for name in names:
                    if is_train:
                        if name not in node2id_global:
                            raise RuntimeError(f"[错误] 训练集出现新节点 {name}")
                        ids.append(node2id_global[name])
                    else:
                        if name in node2id_global:
                            ids.append(node2id_global[name])
                        else:
                            ids.append(temp_counter)
                            temp_counter += 1
                
                ids = torch.tensor(ids, dtype=torch.float32, device=device)
                id_feat[idx] = ids.unsqueeze(1)

            g.ndata["feat"] = torch.cat([orig_feat, id_feat], dim=1)

    add_id_as_feature(train_graphs, train_meta, is_train=True)

    for g, meta in zip(test_graphs, test_meta):
        add_id_as_feature([g], [meta], is_train=False, start_counter=global_id_counter)

    train_num_nodes = sum([g.num_nodes() for g in train_graphs])
    
    return node2id_global, train_graphs, test_graphs


def build_id_maps_and_features_with_rank_optimized(all_graphs,
                                                   feat_key='feat', type_key='_TYPE',
                                                   dim_id=8, num_types=3, embed_dim=64):

    
    eps = 1e-6
    
    def process_graphs(graphs):
        feats_proc = []
        
        for g in graphs:
            feat = g.ndata[feat_key].float()        # [num_nodes,1]
            node_type = g.ndata[type_key]           # [num_nodes]

            type_scale = {0: 100, 1: 1, 2: 100}
            scaled_feat = torch.zeros_like(feat)
            types_in_graph = torch.unique(node_type)
            for t in types_in_graph:
                mask = (node_type == t)
                scaled_feat[mask] = feat[mask] * type_scale[int(t)]
            feat = scaled_feat

            log_feat = torch.log(feat + eps)

            feat_norm = torch.zeros_like(feat)
            rank_feat = torch.zeros_like(feat)
            log_centered = torch.zeros_like(log_feat)

            for t in types_in_graph:
                mask = (node_type == t)
                f = feat[mask]

                feat_norm[mask] = (f - f.mean()) / (f.std() + eps)

                ranks = torch.argsort(torch.argsort(f.view(-1)))
                rank_feat[mask] = ranks.float().unsqueeze(1) / (len(f) - 1 + eps)

                lf = log_feat[mask]
                log_centered[mask] = lf - lf.mean()
            feat_multi = torch.cat([feat_norm,log_centered,rank_feat], dim=1)  # [num_nodes, 4]
            type_scalar = node_type.float()
            type_scalar = type_scalar.unsqueeze(1)
         
           

            feat_final = torch.cat([feat_multi,type_scalar], dim=1)
            
            feats_proc.append(feat_final)
        
        return feats_proc

    all_feat = process_graphs(all_graphs)
    
    return all_feat

def set_seed(seed=66):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.seed(seed)
    except Exception as e:
        print(f"==================================================")
        print(f"⚠️ 警告: DGL种子初始化失败！大概率是当前GPU显存已被占满！")
        print(f"底层报错: {e}")
        print(f"系统将尝试强行继续运行，如果接下来立刻报 OOM 错误，请检查显卡！")
        print(f"==================================================")
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_tensor(x, name):
    if not torch.isfinite(x).all():
        print(f"[NaN DETECTED] {name}")
        print("min:", x.min().item(), "max:", x.max().item())
        raise ValueError(f"{name} contains NaN or Inf")
def extract_topk_attention(model, graph, mp_graphs_list, meta, top_k):

    model.eval()

    with torch.no_grad():

        _,_, att_dict = model(
            graph,
            mp_graphs_list=mp_graphs_list,
            return_attn=True,
        )

    ntype_tensor = graph.ndata[dgl.NTYPE]  # 每个节点的类型
    nid_tensor   = graph.ndata[dgl.NID]    # 每个节点在原异构图中的局部ID

    type_id_to_name = {
        0: "gene",
        1: "microbe",
        2: "pathway"
    }

    result = {}

    for t, type_name in type_id_to_name.items():

        att_scores = att_dict.get(t, None)

        if att_scores is None:
            continue

        att_scores = att_scores.detach().cpu()

        mask = (ntype_tensor == t)
        homo_indices = mask.nonzero(as_tuple=True)[0]

        if len(homo_indices) == 0:
            continue

        assert len(att_scores) == len(homo_indices), \
            f"[错误] 类型 {type_name} attention 数量不匹配: " \
            f"{len(att_scores)} vs {len(homo_indices)}"

        k = min(top_k, len(att_scores))
        top_indices_local = torch.topk(att_scores, k).indices

        selected_homo_indices = homo_indices[top_indices_local]

        selected_orig_ids = nid_tensor[selected_homo_indices]

        names_list = meta[type_name]

        top_pairs = []

        for orig_id, local_idx in zip(selected_orig_ids, top_indices_local):

            orig_id = int(orig_id)
            score   = float(att_scores[local_idx])

            assert orig_id < len(names_list), \
                f"[错误] {type_name} orig_id 越界: {orig_id} >= {len(names_list)}"

            name = names_list[orig_id]

            top_pairs.append((name, score))

        result[type_name] = top_pairs

    return result
