"""Build per-sample heterogeneous graphs for MGPAN preprocessing."""

import glob
import os

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def find_edge_file_flexible(directory, sample_id, patterns):
    """Return the first edge file that matches one of the accepted suffixes."""
    for suffix in patterns:
        pattern = os.path.join(directory, f"{sample_id}_*{suffix}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def build_sample_graph_fast(
    sample_id,
    mm_df,
    pp_df,
    mgp_sample,
    microbe_abundance_df,
    pathway_abundance_df,
    gene_abundance_df=None,
):
    """Build one microbe-gene-pathway heterogeneous graph."""
    all_microbes = sorted(
        set(mm_df["OTU1_gs"])
        | set(mm_df["OTU2_gs"])
        | set(mgp_sample["Taxonomy"])
    )
    all_genes = sorted(mgp_sample["gene"].unique())
    all_pathways = sorted(
        set(pp_df["OTU1_clean"])
        | set(pp_df["OTU2_clean"])
        | set(mgp_sample["pathway"])
    )

    microbe_id = {m: i for i, m in enumerate(all_microbes)}
    gene_id = {g: i for i, g in enumerate(all_genes)}
    pathway_id = {p: i for i, p in enumerate(all_pathways)}

    mm_df = mm_df.drop_duplicates(subset=["OTU1_gs", "OTU2_gs"])
    mm_u = mm_df["OTU1_gs"].map(microbe_id).to_numpy(np.int64)
    mm_v = mm_df["OTU2_gs"].map(microbe_id).to_numpy(np.int64)
    mm_w = (
        mm_df["weight"].to_numpy(np.float32)
        if "weight" in mm_df
        else np.ones(len(mm_u), np.float32)
    )
    mm_src_np = np.concatenate([mm_u, mm_v])
    mm_dst_np = np.concatenate([mm_v, mm_u])
    mm_wt_np = np.concatenate([mm_w, mm_w])

    pp_df = pp_df.drop_duplicates(subset=["OTU1_clean", "OTU2_clean"])
    pp_u = pp_df["OTU1_clean"].map(pathway_id).to_numpy(np.int64)
    pp_v = pp_df["OTU2_clean"].map(pathway_id).to_numpy(np.int64)
    pp_w = (
        pp_df["weight"].to_numpy(np.float32)
        if "weight" in pp_df
        else np.ones(len(pp_u), np.float32)
    )
    pp_src_np = np.concatenate([pp_u, pp_v])
    pp_dst_np = np.concatenate([pp_v, pp_u])
    pp_wt_np = np.concatenate([pp_w, pp_w])

    mg_df = mgp_sample[["Taxonomy", "gene", "weight_mg"]].drop_duplicates()
    mg_src_np = mg_df["Taxonomy"].map(microbe_id).to_numpy(np.int64)
    mg_dst_np = mg_df["gene"].map(gene_id).to_numpy(np.int64)
    mg_wt_np = mg_df["weight_mg"].to_numpy(np.float32)

    gm_df = mgp_sample[["gene", "Taxonomy", "weight_gm"]].drop_duplicates()
    gm_src_np = gm_df["gene"].map(gene_id).to_numpy(np.int64)
    gm_dst_np = gm_df["Taxonomy"].map(microbe_id).to_numpy(np.int64)
    gm_wt_np = gm_df["weight_gm"].to_numpy(np.float32)

    gp_df = mgp_sample[["gene", "pathway", "weight_gp"]].drop_duplicates()
    gp_src_np = gp_df["gene"].map(gene_id).to_numpy(np.int64)
    gp_dst_np = gp_df["pathway"].map(pathway_id).to_numpy(np.int64)
    gp_wt_np = gp_df["weight_gp"].to_numpy(np.float32)

    pg_df = mgp_sample[["pathway", "gene", "weight_pg"]].drop_duplicates()
    pg_src_np = pg_df["pathway"].map(pathway_id).to_numpy(np.int64)
    pg_dst_np = pg_df["gene"].map(gene_id).to_numpy(np.int64)
    pg_wt_np = pg_df["weight_pg"].to_numpy(np.float32)

    edge_dict = {
        ("microbe", "MM", "microbe"): (
            torch.from_numpy(mm_src_np),
            torch.from_numpy(mm_dst_np),
        ),
        ("pathway", "PP", "pathway"): (
            torch.from_numpy(pp_src_np),
            torch.from_numpy(pp_dst_np),
        ),
        ("microbe", "MG", "gene"): (
            torch.from_numpy(mg_src_np),
            torch.from_numpy(mg_dst_np),
        ),
        ("gene", "GM", "microbe"): (
            torch.from_numpy(gm_src_np),
            torch.from_numpy(gm_dst_np),
        ),
        ("gene", "GP", "pathway"): (
            torch.from_numpy(gp_src_np),
            torch.from_numpy(gp_dst_np),
        ),
        ("pathway", "PG", "gene"): (
            torch.from_numpy(pg_src_np),
            torch.from_numpy(pg_dst_np),
        ),
    }

    num_nodes_dict = {
        "microbe": len(all_microbes),
        "gene": len(all_genes),
        "pathway": len(all_pathways),
    }

    graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)

    microbe_feats = torch.tensor(
        [
            microbe_abundance_df.loc[mid, sample_id]
            if mid in microbe_abundance_df.index
            else 0.0
            for mid in all_microbes
        ],
        dtype=torch.float32,
    ).unsqueeze(1)

    pathway_feats = torch.tensor(
        [
            pathway_abundance_df.loc[pid, sample_id]
            if pid in pathway_abundance_df.index
            else 0.0
            for pid in all_pathways
        ],
        dtype=torch.float32,
    ).unsqueeze(1)

    gene_map = (
        mgp_sample.set_index("gene")["GeneAbundance"].to_dict()
        if "GeneAbundance" in mgp_sample.columns
        else {}
    )
    gene_feats = torch.tensor(
        [gene_map.get(gene, 0.0) for gene in all_genes],
        dtype=torch.float32,
    ).unsqueeze(1)

    graph.nodes["microbe"].data["feat"] = microbe_feats
    graph.nodes["gene"].data["feat"] = gene_feats
    graph.nodes["pathway"].data["feat"] = pathway_feats

    graph.edges["MM"].data["weight"] = torch.from_numpy(mm_wt_np)
    graph.edges["PP"].data["weight"] = torch.from_numpy(pp_wt_np)
    graph.edges["MG"].data["weight"] = torch.from_numpy(mg_wt_np)
    graph.edges["GM"].data["weight"] = torch.from_numpy(gm_wt_np)
    graph.edges["GP"].data["weight"] = torch.from_numpy(gp_wt_np)
    graph.edges["PG"].data["weight"] = torch.from_numpy(pg_wt_np)

    graph.microbe_names = all_microbes
    graph.gene_names = all_genes
    graph.pathway_names = all_pathways

    return graph


class GraphDataset(Dataset):
    """Build and store all sample graphs for preprocessing."""

    def __init__(
        self,
        sample_ids,
        labels,
        microbe_edges_dir,
        pathway_edges_dir,
        mgp_by_sample,
        microbe_abundance_df,
        pathway_abundance_df,
        gene_abundance_df,
    ):
        self.sample_ids = sample_ids
        self.labels = labels
        self.microbe_edges_dir = microbe_edges_dir
        self.pathway_edges_dir = pathway_edges_dir
        self.mgp_by_sample = mgp_by_sample
        self.microbe_abundance_df = microbe_abundance_df
        self.pathway_abundance_df = pathway_abundance_df
        self.gene_abundance_df = gene_abundance_df

        self.graphs = [
            build_sample_graph_fast(
                sid,
                mm_df=pd.read_csv(
                    find_edge_file_flexible(
                        microbe_edges_dir,
                        sid,
                        ["MM_edges.csv", "microbe_edges.csv"],
                    )
                ),
                pp_df=pd.read_csv(
                    find_edge_file_flexible(
                        pathway_edges_dir,
                        sid,
                        ["PP_edges.csv", "pathway_edges.csv"],
                    )
                ),
                mgp_sample=self.mgp_by_sample[sid],
                microbe_abundance_df=self.microbe_abundance_df,
                pathway_abundance_df=self.pathway_abundance_df,
                gene_abundance_df=self.gene_abundance_df,
            )
            for sid in self.sample_ids
        ]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
