# MGPAN

## English

MGPAN is a meta-path guided graph neural network for graph-level classification on heterogeneous microbiome-related graphs.

## 中文

MGPAN 是一个面向异质微生物组相关图的 Meta-path guided graph neural network，用于图级分类任务。

## Project Structure / 项目结构

```text
MGPAN/
|-- README.md
|-- requirements.txt
|-- config.py
|-- main.py
|-- model/
|   |-- __init__.py
|   |-- mgpan.py
|   `-- trainer.py
|-- utils/
|   |-- __init__.py
|   |-- data_loader.py
|   `-- metrics.py
`-- data_process/
    |-- README.md
    |-- 01_prepare_mgp_inputs_and_edges.ipynb
    |-- 02_build_graph_dataset.ipynb
    |-- graph_dataset.py
    `-- prepare_mgp_config.py
```

## Environment / 环境配置

### English

Create an environment and install dependencies from the repository root:

```bash
conda create -n mgpan python=3.8 -y
conda activate mgpan
pip install -r requirements.txt
```

If you use the original CUDA/DGL environment from the paper experiments, install the matching PyTorch and DGL versions listed in `requirements.txt`.

### 中文

在仓库根目录下创建环境并安装依赖：

```bash
conda create -n mgpan python=3.8 -y
conda activate mgpan
pip install -r requirements.txt
```

如果你希望复现论文实验中的 CUDA/DGL 环境，请按照 `requirements.txt` 中列出的 PyTorch 和 DGL 版本安装。

## Data Preparation / 数据准备

### English

Input abundance files are expected under:

```text
MGPAN/data/<dataset>/
```

For example:

```text
MGPAN/data/QinN_2014/gene_families_abundance.csv
MGPAN/data/QinN_2014/pathway_abundance_abundance.csv
MGPAN/data/QinN_2014/relative_abundance_abundance.csv
```

The preprocessing notebook uses `data_process/prepare_mgp_config.py` to locate the data. You can change the dataset without editing notebook cells by setting environment variables:

```bash
export MGP_DATA_ROOT=/path/to/MGPAN/data
export MGP_HUMANN_DATASET=QinN_2014
export MGP_EDGE_DATASET=QinN_2014
```

On Windows CMD:

```bat
set MGP_DATA_ROOT=D:\path\to\MGPAN\data
set MGP_HUMANN_DATASET=QinN_2014
set MGP_EDGE_DATASET=QinN_2014
```

Before running the full preprocessing notebook, generate `path_taxonomy_uf90.tsv` with HUMAnN's `merge_abundance.py`. See the detailed guide:

```text
data_process/README.md
```

### 中文

输入丰度文件默认放在：

```text
MGPAN/data/<dataset>/
```

例如：

```text
MGPAN/data/QinN_2014/gene_families_abundance.csv
MGPAN/data/QinN_2014/pathway_abundance_abundance.csv
MGPAN/data/QinN_2014/relative_abundance_abundance.csv
```

预处理 notebook 会通过 `data_process/prepare_mgp_config.py` 统一读取路径。你可以不修改 notebook 代码，而是通过环境变量切换数据集：

```bash
export MGP_DATA_ROOT=/path/to/MGPAN/data
export MGP_HUMANN_DATASET=QinN_2014
export MGP_EDGE_DATASET=QinN_2014
```

Windows CMD：

```bat
set MGP_DATA_ROOT=D:\path\to\MGPAN\data
set MGP_HUMANN_DATASET=QinN_2014
set MGP_EDGE_DATASET=QinN_2014
```

在运行完整预处理 notebook 前，需要先用 HUMAnN 的 `merge_abundance.py` 生成 `path_taxonomy_uf90.tsv`。详细教程见：

```text
data_process/README.md
```

## Reproduction Workflow / 复现流程

### English

Run the following commands from the repository root.

1. Prepare HUMAnN merged abundance input:

```bash
jupyter notebook data_process/01_prepare_mgp_inputs_and_edges.ipynb
```

2. Build graph data:

```bash
jupyter notebook data_process/02_build_graph_dataset.ipynb
```

3. Train and evaluate MGPAN:

```bash
python main.py
```

Common overrides:

```bash
python main.py \
  --dataset NielsenHB_2014 \
  --graphdata NH_graphdataF1.bin \
  --metadata NH_graphdata_metaF1.pkl \
  --epochs 100 \
  --batch-size 64
```

Meta-path graph caches are written to `metapaths/<dataset>/MGPAN_metapath_graphs8/`, logs to `logs/<dataset>/metapaths/`, and figures/raw predictions to `data/<dataset>/figures_<log>/`.

### 中文

在仓库根目录下依次执行以下步骤。

1. 准备 HUMAnN 合并丰度输入：

```bash
jupyter notebook data_process/01_prepare_mgp_inputs_and_edges.ipynb
```

2. 构建图数据：

```bash
jupyter notebook data_process/02_build_graph_dataset.ipynb
```

3. 训练并评估 MGPAN：

```bash
python main.py
```

常用参数覆盖示例：

```bash
python main.py \
  --dataset NielsenHB_2014 \
  --graphdata NH_graphdataF1.bin \
  --metadata NH_graphdata_metaF1.pkl \
  --epochs 100 \
  --batch-size 64
```

Meta-path 图缓存会写入 `metapaths/<dataset>/MGPAN_metapath_graphs8/`，日志会写入 `logs/<dataset>/metapaths/`，图像和原始预测结果会写入 `data/<dataset>/figures_<log>/`。

## Notes / 说明

### English

The refactored package keeps the original training objective, eight fixed meta-path definitions, graph augmentation, node-type-aware readout, and cross-validation workflow from `main_metapath_Gp.py`. The old model name has been replaced by `MGPAN` throughout the package.

### 中文

重构后的代码保留了 `main_metapath_Gp.py` 中的原始训练目标、8 条固定 meta-path 定义、图增强、node-type-aware readout 和交叉验证流程。旧模型名称已经在整理后的包中统一替换为 `MGPAN`。
