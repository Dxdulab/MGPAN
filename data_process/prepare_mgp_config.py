"""Centralized paths and parameters for MGP input preparation."""

from __future__ import annotations

import os
from pathlib import Path


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _resolve_path(value, base_dir: Path) -> Path:
    path = Path(str(value).strip().strip("\"'")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _env_path(name: str, default, base_dir: Path) -> Path:
    value = os.getenv(name, "").strip()
    return _resolve_path(value or default, base_dir)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = _env_path("MGP_DATA_ROOT", PACKAGE_ROOT / "data", PROJECT_ROOT)

HUMANN_DATASET = _env_text("MGP_HUMANN_DATASET", "QinN_2014")
EDGE_DATASET = _env_text("MGP_EDGE_DATASET", HUMANN_DATASET)

HUMANN_DIR = _resolve_path(DATA_ROOT / HUMANN_DATASET, PROJECT_ROOT)
EDGE_DIR = _resolve_path(DATA_ROOT / EDGE_DATASET, PROJECT_ROOT)

GENE_FAMILIES_CSV = HUMANN_DIR / "gene_families_abundance.csv"
GENE_FAMILIES_TSV = HUMANN_DIR / "gene_families_abundance.tsv"
PATHWAY_ABUNDANCE_CSV = HUMANN_DIR / "pathway_abundance_abundance.csv"
PATHWAY_ABUNDANCE_TSV = HUMANN_DIR / "pathway_abundance_abundance.tsv"
MICROBE_ABUNDANCE_CSV = HUMANN_DIR / "relative_abundance_abundance.csv"

PATH_TAXONOMY_TSV = HUMANN_DIR / "path_taxonomy_uf90.tsv"
PATH_TAXONOMY_CSV = HUMANN_DIR / "path-tax-uf90.csv"
SAMPLE_PATH_TAX_GENE_CSV = HUMANN_DIR / "sample_path_tax_gene.csv"
SAMPLE_PATH_TAX_GENE_WEIGHT_CSV = HUMANN_DIR / "sample_path_tax_gene_weight.csv"

MICROBE_ABUNDANCE_CLEANED_CSV = HUMANN_DIR / "relative_abundance_cleaned1.csv"
PATHWAY_ABUNDANCE_CLEANED_CSV = HUMANN_DIR / "pathway_abundance_cleaned1.csv"
MICROBE_NAME_MAPPING_CSV = HUMANN_DIR / "microbe_name_mapping.csv"
PATHWAY_NAME_MAPPING_CSV = HUMANN_DIR / "pathway_name_mapping.csv"

METADATA_CSV = HUMANN_DIR / "relative_abundance_metadata.csv"
METADATA_CRC_CSV = HUMANN_DIR / "relative_abundance_metadata_CRC.csv"
SUBJECT_IDS_CSV = HUMANN_DIR / "subject_ids.csv"

PP_INPUT_CSV = EDGE_DIR / "pathway_abundance_abundance.csv"
PP_OUTPUT_DIR = EDGE_DIR / "pathway"
MM_INPUT_CSV = EDGE_DIR / "relative_abundance_abundance.csv"
MM_OUTPUT_DIR = EDGE_DIR / "microbe"

DATASET_GRAPH_PREFIXES = {
    "QinN_2014": "QN",
    "NielsenHB_2014": "NH",
    "LifeLD_VilaAV_2018": "LL",
}
GRAPH_PREFIX = _env_text(
    "MGP_GRAPH_PREFIX",
    DATASET_GRAPH_PREFIXES.get(EDGE_DATASET, EDGE_DATASET),
)
GRAPH_RAW_BIN = HUMANN_DIR / f"{GRAPH_PREFIX}_graphdata.bin"
GRAPH_RAW_META_PKL = HUMANN_DIR / f"{GRAPH_PREFIX}_graphdata_meta.pkl"
GRAPH_PRUNED_BIN = HUMANN_DIR / f"{GRAPH_PREFIX}_graphdataF6.bin"
GRAPH_PRUNED_META_PKL = HUMANN_DIR / f"{GRAPH_PREFIX}_graphdata_metaF6.pkl"
FEATURE_VOCAB_PKL = HUMANN_DIR / "feature_vocab.pkl"

LABEL_COL = _env_text("MGP_LABEL_COL", "study_condition")
SAMPLE_ID_COL = _env_text("MGP_SAMPLE_ID_COL", "sample_id")
SUBJECT_ID_COL = _env_text("MGP_SUBJECT_ID_COL", "subject_id")

WEIGHT_EPS = 1e-8
EDGE_EPS = 1e-6
TRIPLET_ABUNDANCE_THRESHOLD = 0

PP_GAMMA = 3.0
PP_TOP_K = 4
PP_ABUNDANCE_THRESHOLD = 0.0002

MM_GAMMA = 3.0
MM_TOP_K = 3
MM_ABUNDANCE_THRESHOLD = 0.01



# Node pruning in 02_build_graph_dataset.ipynb keeps top-K abundant nodes per type.
GRAPH_PRUNE_ABUNDANCE_TOP_K = {
    "microbe": 140,
    "gene": 120,
    "pathway": 120,
}

REQUIRED_INPUT_FILES = {
    "gene families": GENE_FAMILIES_CSV,
    "pathway abundance": PATHWAY_ABUNDANCE_CSV,
    "microbe abundance": MICROBE_ABUNDANCE_CSV,
    "P-P edge input": PP_INPUT_CSV,
    "M-M edge input": MM_INPUT_CSV,
}

GRAPH_INPUT_FILES = {
    "cleaned microbe abundance": MICROBE_ABUNDANCE_CLEANED_CSV,
    "cleaned pathway abundance": PATHWAY_ABUNDANCE_CLEANED_CSV,
    "gene families": GENE_FAMILIES_CSV,
    "sample-path-tax-gene weights": SAMPLE_PATH_TAX_GENE_WEIGHT_CSV,
    "metadata": METADATA_CSV,
}

OUTPUT_DIRS = (PP_OUTPUT_DIR, MM_OUTPUT_DIR)


def ensure_output_dirs() -> None:
    for output_dir in OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)


def _validate_file_map(file_map) -> None:
    missing = {
        label: path
        for label, path in file_map.items()
        if not path.exists()
    }
    if not missing:
        return

    lines = ["Missing required MGP input files:"]
    lines.extend(f"- {label}: {path}" for label, path in missing.items())
    lines.append(
        "Set MGP_DATA_ROOT, MGP_HUMANN_DATASET, or MGP_EDGE_DATASET "
        "if your data are stored elsewhere."
    )
    raise FileNotFoundError("\n".join(lines))


def validate_inputs() -> None:
    _validate_file_map(REQUIRED_INPUT_FILES)


def validate_graph_inputs() -> None:
    _validate_file_map(GRAPH_INPUT_FILES)
    missing_dirs = [
        path
        for path in (MM_OUTPUT_DIR, PP_OUTPUT_DIR)
        if not path.exists()
    ]
    if missing_dirs:
        lines = ["Missing required edge output directories:"]
        lines.extend(f"- {path}" for path in missing_dirs)
        lines.append("Run 01_prepare_mgp_inputs_and_edges.ipynb before this step.")
        raise FileNotFoundError("\n".join(lines))


def print_config(validate: bool = True) -> None:
    if validate:
        validate_inputs()
    ensure_output_dirs()

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"PACKAGE_ROOT: {PACKAGE_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"HUMAnN dataset: {HUMANN_DATASET} -> {HUMANN_DIR}")
    print(f"Edge dataset: {EDGE_DATASET} -> {EDGE_DIR}")
    print(f"P-P input: {PP_INPUT_CSV}")
    print(f"M-M input: {MM_INPUT_CSV}")
    print(f"P-P output dir: {PP_OUTPUT_DIR}")
    print(f"M-M output dir: {MM_OUTPUT_DIR}")
    print(f"Metadata: {METADATA_CSV}")
    print(f"Subject IDs: {SUBJECT_IDS_CSV}")
    print(f"Raw graph: {GRAPH_RAW_BIN}")
    print(f"Raw graph metadata: {GRAPH_RAW_META_PKL}")
    print(f"Pruned graph: {GRAPH_PRUNED_BIN}")
    print(f"Pruned graph metadata: {GRAPH_PRUNED_META_PKL}")
