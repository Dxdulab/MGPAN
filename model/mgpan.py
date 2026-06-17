"""Core MGPAN model definition."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GINConv, SAGEConv
from torch import nn
from torch.utils.data import DataLoader


class MGPAN(nn.Module):
    """Meta-path guided and type-aware graph classifier."""

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
    ):
        super().__init__()
        self.num_node_ids = num_node_ids
        self.dropout1 = dropout1

        self.embedder = MGPANGraph(
            gnn_type=gnn_type,
            num_gnn_layers=num_gnn_layers,
            relations=relations,
            feat_dim=embed_dim,
            embed_dim=embed_dim,
            dim_a=dim_a,
            dropout2=dropout2,
            dropout3=dropout3,
            attdropout=attdropout,
            activation=activation.casefold(),
            metapaths=metapaths,
        )

        hidden_dim = 8
        self.type_emb = nn.Embedding(num_node_types, type_emb_dim)
        self.type_emb_mlp = nn.Sequential(
            nn.Linear(type_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, type_emb_dim),
        )
        self.node_id_emb = nn.Embedding(num_node_ids + 1, node_id_emb_dim)
        self.node_id_emb_mlp = nn.Sequential(
            nn.Linear(node_id_emb_dim, node_id_emb_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout1),
        )
        self.abundance_proj = nn.Sequential(
            nn.Linear(3, abundance_proj_dim),
            nn.ReLU(),
            nn.Linear(abundance_proj_dim, abundance_proj_dim),
            nn.ReLU(),
        )

        total_input_dim = abundance_proj_dim + type_emb_dim + node_id_emb_dim
        self.node_feat_proj = nn.Linear(total_input_dim, embed_dim)
        self.node_feat_norm = nn.LayerNorm(embed_dim)
        self.classifier = MinimalClassifier(embed_dim=embed_dim, dropout=0.45)

    def forward(self, graph, mp_graphs_list=None, return_attn=False):
        feat = graph.ndata["feat"].float()
        check_tensor(feat, "raw feat")

        abundance_feats = self.abundance_proj(feat[:, :3])
        check_tensor(abundance_feats, "abundance_feats")

        node_type = feat[:, 3].long()
        type_feats = self.type_emb_mlp(self.type_emb(node_type))

        node_ids = feat[:, 4].long()
        unk_id = self.num_node_ids
        node_ids = torch.where(
            node_ids >= unk_id,
            node_ids.new_full(node_ids.shape, unk_id),
            node_ids,
        )
        node_id_feats = self.node_id_emb_mlp(self.node_id_emb(node_ids))
        check_tensor(node_id_feats, "node_id_feats")

        h = torch.cat([abundance_feats, type_feats, node_id_feats], dim=1)
        h = self.node_feat_proj(h)
        check_tensor(h, "node_feat_proj")
        h = self.node_feat_norm(h)
        check_tensor(h, "node_feat_norm")
        h = F.relu(h)
        check_tensor(h, "relu h")
        h = F.dropout(h, p=self.dropout1, training=self.training)

        if return_attn:
            embed, att_dict = self.embedder(
                graph,
                h,
                return_attn=True,
                mp_graphs_list=mp_graphs_list,
            )
            out = self.classifier(embed)
            return out, embed, att_dict

        embed = self.embedder(graph, h, mp_graphs_list=mp_graphs_list)
        out = self.classifier(embed)
        return out, embed


class MGPANGraph(nn.Module):
    """Node-level MGPAN encoder followed by type-aware graph pooling."""

    def __init__(
        self,
        gnn_type,
        num_gnn_layers,
        relations,
        feat_dim,
        embed_dim,
        dim_a,
        dropout2=0.0,
        dropout3=0.0,
        attdropout=0.0,
        activation=None,
        metapaths=None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            MGPANLayer(
                gnn_type=gnn_type,
                relations=relations,
                in_dim=feat_dim,
                out_dim=embed_dim,
                dim_a=dim_a,
                attdropout=attdropout,
                dropout3=dropout3,
                activation=activation,
                metapaths=metapaths,
            )
        )
        for _ in range(1, num_gnn_layers):
            self.layers.append(
                MGPANLayer(
                    gnn_type=gnn_type,
                    relations=relations,
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    dim_a=dim_a,
                    attdropout=attdropout,
                    dropout3=dropout3,
                    activation=activation,
                    metapaths=metapaths,
                )
            )

        self.type_pool = NodeTypeAwarePooling(
            embed_dim=embed_dim,
            att_hidden_dim=64,
            att_dropout=attdropout,
        )
        self.graph_feat_proj = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, graph, feat, return_attn=False, mp_graphs_list=None):
        h = feat
        for layer in self.layers:
            h = layer(graph, h, mp_graphs_list=mp_graphs_list)

        if return_attn:
            h_readout, att_dict = self.type_pool(graph, h, return_attn=True)
        else:
            h_readout = self.type_pool(graph, h)

        h_readout = F.relu(self.graph_feat_proj(h_readout))
        if return_attn:
            return h_readout, att_dict
        return h_readout


class MGPANLayer(nn.Module):
    """One MGPAN layer over fixed meta-path graphs."""

    def __init__(
        self,
        gnn_type,
        relations,
        in_dim,
        out_dim,
        dim_a,
        attdropout,
        dropout3=0.0,
        activation="relu",
        metapaths=None,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.relations = relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout3 = dropout3
        self.attdropout = attdropout
        self.activation = self._get_activation_fn(activation)
        self.num_metapaths = len(metapaths)

        self.mp_layers = nn.ModuleList()
        for _ in range(self.num_metapaths):
            if self.gnn_type == "gin":
                conv = GINConv(
                    apply_func=nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.Dropout(p=self.dropout3),
                        self.activation,
                        nn.Linear(out_dim, out_dim),
                        nn.Dropout(p=self.dropout3),
                        self.activation,
                    ),
                    aggregator_type="sum",
                )
            elif self.gnn_type == "gatv2":
                conv = GATv2ConvEdgeOnly(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    num_heads=1,
                    feat_drop=self.dropout3,
                    attn_drop=self.dropout3,
                    residual=False,
                    activation=self.activation,
                    allow_zero_in_degree=True,
                )
            elif self.gnn_type == "sage":
                conv = SAGEConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    aggregator_type="pool",
                    feat_drop=self.dropout3,
                    activation=self.activation,
                )
            else:
                raise ValueError(
                    f"Invalid gnn_type: {gnn_type}. Choose 'gin', 'gatv2', or 'sage'."
                )
            self.mp_layers.append(conv)


        self.attention = MetaPathAttention(
            num_metapaths=self.num_metapaths,
            embed_dim=out_dim,
            dim_a=self.dim_a,
            dropout=self.attdropout,
        )

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            return None
        if activation == "relu":
            return nn.ReLU()
        if activation == "elu":
            return nn.ELU()
        if activation == "gelu":
            return nn.GELU()
        raise ValueError(f"Invalid activation function: {activation}")

    @staticmethod
    def _align_batched_features(feat, batched_mp_graph):
        if "_ID" not in batched_mp_graph.ndata:
            return feat

        local_ids = batched_mp_graph.ndata["_ID"].to(feat.device)
        batch_num_nodes = batched_mp_graph.batch_num_nodes().to(feat.device)
        offsets = torch.cumsum(
            torch.cat(
                [
                    batch_num_nodes.new_zeros(1),
                    batch_num_nodes[:-1],
                ]
            ),
            dim=0,
        )
        offsets = torch.repeat_interleave(offsets, batch_num_nodes)
        return feat[local_ids + offsets]

    def forward(self, graph, feat, mp_graphs_list, device=None):
        if mp_graphs_list is None:
            raise ValueError("mp_graphs_list is required for MGPAN.")

        if device is None:
            device = graph.device

        mp_node_embs = []
        for path_idx, gnn_layer in enumerate(self.mp_layers):
            graphs_i = [
                sample_graphs[path_idx].to(device)
                for sample_graphs in mp_graphs_list
            ]
            batched_mp_graph = dgl.batch(graphs_i)
            feat_mp = self._align_batched_features(feat, batched_mp_graph)

            if feat_mp.shape[0] != batched_mp_graph.num_nodes():
                raise RuntimeError(
                    f"[Metapath {path_idx}] feature mismatch: "
                    f"{feat_mp.shape[0]} vs {batched_mp_graph.num_nodes()}"
                )

            feat_mp = F.dropout(feat_mp, p=self.dropout3, training=self.training)
            edge_weight = batched_mp_graph.edata.get("weight", None)
            if edge_weight is not None and not torch.isfinite(edge_weight).all():
                raise RuntimeError(
                    f"[Metapath {path_idx}] edge_weight contains NaN or Inf."
                )

            if isinstance(gnn_layer, GATv2ConvEdgeOnly):
                h_out = gnn_layer(
                    batched_mp_graph,
                    feat_mp,
                    edge_weight=edge_weight,
                )
                if h_out.dim() == 3:
                    h_out = h_out.mean(dim=1)
            else:
                h_out = gnn_layer(
                    batched_mp_graph,
                    feat_mp,
                    edge_weight=edge_weight,
                )

            mp_node_embs.append(h_out.unsqueeze(0))

        h_views = torch.cat(mp_node_embs, dim=0)
        h_views = F.layer_norm(h_views, h_views.shape[-1:])
        h_views = F.dropout(h_views, p=self.dropout3, training=self.training)

        fused_h = self.attention(h_views)
        h_out=fused_h + feat
        return h_out


class MetaPathAttention(nn.Module):
    """Node-level attention over a fixed set of meta-path views."""

    def __init__(
        self,
        num_metapaths: int,
        embed_dim: int,
        dim_a: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)
        self.weights_s1 = nn.Parameter(torch.empty(num_metapaths, embed_dim, dim_a))
        self.weights_s2 = nn.Parameter(torch.empty(num_metapaths, dim_a, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("tanh")
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(
        self,
        h_views: torch.Tensor,
        batch_size: int = 32,
        return_alpha: bool = False,
    ):
        num_metapaths, num_nodes, embed_dim = h_views.shape
        fused_h = torch.zeros(num_nodes, embed_dim, device=h_views.device)
        alpha_sum = torch.zeros(num_metapaths, device=h_views.device)
        node_count = 0

        node_loader = DataLoader(
            list(range(num_nodes)),
            batch_size=batch_size,
            shuffle=False,
        )

        for node_batch in node_loader:
            h_batch = h_views[:, node_batch, :]
            alpha = torch.matmul(
                torch.tanh(torch.matmul(h_batch, self.weights_s1)),
                self.weights_s2,
            )
            alpha = F.softmax(alpha, dim=0).squeeze(-1)
            alpha = self.dropout(alpha)

            fused_h[node_batch] = torch.einsum("rb,rbd->bd", alpha, h_batch)
            alpha_sum += alpha.sum(dim=1)
            node_count += alpha.shape[1]

        if return_alpha:
            return fused_h, alpha_sum / max(node_count, 1)

        return fused_h


class NodeTypeAwarePooling(nn.Module):
    """Graph readout that pools nodes separately for each node type."""

    def __init__(
        self,
        embed_dim: int,
        att_hidden_dim: int = 64,
        att_dropout: float = 0.35,
        num_types: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_types = num_types
        self.att_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, att_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(att_dropout),
                    nn.Linear(att_hidden_dim, 1),
                )
                for _ in range(num_types)
            ]
        )

    @staticmethod
    def _group_softmax(scores, graph_ids, num_graphs):
        alpha = torch.zeros_like(scores)
        for graph_id in range(num_graphs):
            mask = graph_ids == graph_id
            if mask.any():
                alpha[mask] = torch.softmax(scores[mask], dim=0)
        return alpha

    def forward(self, graph, h, return_attn: bool = False):
        ntype_tensor = graph.ndata[dgl.NTYPE].to(h.device)
        batch_num_nodes = graph.batch_num_nodes().to(h.device)
        num_graphs = batch_num_nodes.shape[0]
        graph_ids = torch.repeat_interleave(
            torch.arange(num_graphs, device=h.device),
            batch_num_nodes,
        )

        type_feats = []
        att_dict = {} if return_attn else None

        for node_t in range(self.num_types):
            type_idx = (ntype_tensor == node_t).nonzero(as_tuple=False).squeeze(-1)
            pooled_t = h.new_zeros(num_graphs, self.embed_dim)

            if type_idx.numel() > 0:
                h_t = h[type_idx]
                graph_ids_t = graph_ids[type_idx]
                scores_t = self.att_gates[node_t](h_t)
                alpha_t = self._group_softmax(scores_t, graph_ids_t, num_graphs)
                pooled_t.index_add_(dim=0, index=graph_ids_t, source=alpha_t * h_t)

                if return_attn:
                    att_dict[node_t] = alpha_t.squeeze(-1)
            elif return_attn:
                att_dict[node_t] = h.new_zeros(0)

            type_feats.append(pooled_t)

        h_graph = torch.cat(type_feats, dim=1)
        if return_attn:
            return h_graph, att_dict
        return h_graph


class GATv2ConvEdgeOnly(nn.Module):
    """GATv2 convolution where edge weights modulate attention scores."""

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        share_weights=False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.share_weights = share_weights
        self.bias = bias

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats,
                out_feats * num_heads,
                bias=bias,
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats,
                out_feats * num_heads,
                bias=bias,
            )
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats,
                out_feats * num_heads,
                bias=bias,
            )
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats,
                    out_feats * num_heads,
                    bias=bias,
                )

        self.attn = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats,
                    num_heads * out_feats,
                    bias=bias,
                )
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_src.bias, 0)

        if not self.share_weights:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.fc_dst.bias, 0)

        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise ValueError("Graph has zero in-degree nodes.")

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(
                    -1,
                    self._num_heads,
                    self._out_feats,
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1,
                    self._num_heads,
                    self._out_feats,
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1,
                    self._num_heads,
                    self._out_feats,
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1,
                    self._num_heads,
                    self._out_feats,
                )
                if graph.is_block:
                    feat_dst = feat_dst[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]

            graph.srcdata["h_src"] = feat_src
            graph.dstdata["h_dst"] = feat_dst
            graph.apply_edges(fn.u_add_v("h_src", "h_dst", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            e = (e * self.attn).sum(dim=-1, keepdim=True)

            if edge_weight is not None:
                e = e * edge_weight.view(-1, 1, 1)

            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e("h_src", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0],
                    -1,
                    self._out_feats,
                )
                rst = rst + resval

            if self.activation is not None:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            return rst


class MinimalClassifier(nn.Module):
    """Final binary classifier on graph-level embeddings."""

    def __init__(self, embed_dim, dropout=0.45):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


def check_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        print(f"[NaN DETECTED] {name}")
        print("min:", x.min().item(), "max:", x.max().item())
        raise ValueError(f"{name} contains NaN or Inf")


def expand_as_pair(input_value):
    if isinstance(input_value, tuple):
        return input_value
    return input_value, input_value
