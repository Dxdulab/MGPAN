"""Attention and edge-aware convolution modules for MGPAN."""

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from torch import nn
from torch.utils.data import DataLoader
from dgl.nn import GlobalAttentionPooling
class MetaPathAttention(nn.Module):
    """
    Meta-path attention for batched nodes.
    输入: h_views (num_metapaths, num_nodes, embed_dim)
    输出: fused_h (num_nodes, embed_dim)
    """

    def __init__(self, num_metapaths, embed_dim, dim_a=64, dropout=0.):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(num_metapaths, embed_dim, dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(num_metapaths, dim_a, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, h_views, batch_size=32, return_alpha=False):
        """
        h_views: (num_metapaths, num_nodes, embed_dim)
        return:
            fused_h: (num_nodes, embed_dim)
            alpha_mean (optional): (num_metapaths,)
        """

        num_metapaths, num_nodes, embed_dim = h_views.shape
        device = h_views.device

        fused_h = torch.zeros(num_nodes, embed_dim, device=device)

        alpha_sum = torch.zeros(num_metapaths, device=device)
        node_count = 0

        node_loader = DataLoader(
            list(range(num_nodes)),
            batch_size=batch_size,
            shuffle=False
        )

        for node_batch in node_loader:
            h_batch = h_views[:, node_batch, :]  # (R, B, D)

            alpha = torch.matmul(
                torch.tanh(torch.matmul(h_batch, self.weights_s1)),
                self.weights_s2
            )  # (R, B, 1)

            alpha = F.softmax(alpha, dim=0).squeeze(-1)  # (R, B)
            alpha = self.dropout(alpha)

            fused_h[node_batch] = torch.einsum(
                'rb,rbd->bd',
                alpha,
                h_batch
            )

            alpha_sum += alpha.sum(dim=1)
            node_count += alpha.shape[1]

        alpha_mean = alpha_sum / node_count  # (num_metapaths,)

        if not self.training:
            print("Metapath attention mean:", alpha_mean.detach().cpu().numpy())

        if return_alpha:
            return fused_h, alpha_mean

        return fused_h


def expand_as_pair(input):
    """
    If input is a tuple, return as-is (src, dst),
    else return (input, input)
    """
    if isinstance(input, tuple):
        return input
    return (input, input)




class NodeTypeAwarePooling(nn.Module):

    def __init__(self, embed_dim, att_hidden_dim=32, att_dropout=0.35, num_types=3):
        super().__init__()
        self.num_types = num_types

        self.att_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, att_hidden_dim),
                nn.ReLU(),
                nn.Dropout(att_dropout),
                nn.Linear(att_hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_types)
        ])

        self.pools = nn.ModuleList([
            GlobalAttentionPooling(gate)
            for gate in self.att_gates
        ])

    def forward(self, graph, h, return_attn=False):

        ntype_tensor = graph.ndata[dgl.NTYPE]
        type_feats = []
        att_dict = {} if return_attn else None

        for t in range(self.num_types):

            mask = (ntype_tensor == t)
            h_masked = torch.zeros_like(h)

            if mask.sum() > 0:
                h_masked[mask] = h[mask]

            if return_attn:
                pooled, att = self.pools[t](
                    graph,
                    h_masked,
                    get_attention=True
                )
                # 只保留该类型节点 attention
                att_dict[t] = att[mask].squeeze()
            else:
                pooled = self.pools[t](graph, h_masked)

            type_feats.append(pooled)

        h_graph = torch.cat(type_feats, dim=1)

        if return_attn:
            return h_graph, att_dict
        else:
            return h_graph


class GATv2ConvEdgeOnly(nn.Module):
    """
    GATv2Conv with optional edge weights applied only to attention.
    Node features are updated, edges are not updated.
    """
    def __init__(self, in_feats, out_feats, num_heads,
                 feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False,
                 activation=None, allow_zero_in_degree=False,
                 bias=True, share_weights=False):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.share_weights = share_weights

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = self.fc_src if share_weights else nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)

        self.attn = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = nn.Identity()
        else:
            self.res_fc = None

        self.activation = activation
        self.bias = bias
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
        """
        feat: (N, in_feats) or tuple (N_src, N_dst)
        edge_weight: (E,) optional, scalar edge weight
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise ValueError("Graph has 0-in-degree nodes. Set allow_zero_in_degree=True to override.")

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = feat_src if self.share_weights else self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_dst[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]

            graph.srcdata['h_src'] = feat_src
            graph.dstdata['h_dst'] = feat_dst

            graph.apply_edges(fn.u_add_v('h_src', 'h_dst', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            e = (e * self.attn).sum(dim=-1, keepdim=True)  # (E, H, 1)

            if edge_weight is not None:
                e = e * edge_weight.view(-1, 1, 1)

            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))

            graph.update_all(fn.u_mul_e('h_src', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            if self.activation is not None:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            return rst
