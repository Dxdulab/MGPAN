# MGPAN

## English

MGPAN is a meta-path guided graph neural network for graph-level classification on heterogeneous microbiome-related graphs. The project builds sample-level heterogeneous graphs from microbiome abundance data and trains MGPAN with fixed meta-path views, graph augmentation, node-type-aware pooling, and cross-validation.

## 中文

MGPAN 是一个面向异质微生物组图的 meta-path guided graph neural network，用于样本级图分类任务。本项目从微生物组丰度数据构建样本级异质图，并使用固定 meta-path 视图、图增强、节点类型感知读出和交叉验证训练 MGPAN。

## Project Structure / 项目结构

```text
MGPAN/
|-- README.md
|-- requirements.txt
|-- config.py
|-- main.py
|-- model/
|   |-- __init__.py
|   |-- attention.py
|   `-- mgpan.py
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

Create an environment and install dependencies from the repository root:

```bash
conda create -n mgpan python=3.9 -y
conda activate mgpan
pip install -r requirements.txt
```

If you use a CUDA/DGL environment, install PyTorch and DGL versions that match your CUDA version and the packages listed in `requirements.txt`.

在仓库根目录下创建环境并安装依赖：

```bash
conda create -n mgpan python=3.9 -y
conda activate mgpan
pip install -r requirements.txt
```

如果使用 CUDA/DGL 环境，请安装与本机 CUDA 版本以及 `requirements.txt` 匹配的 PyTorch 和 DGL。

## Data Preparation / 数据准备

Input abundance files are expected under:

```text
data/<dataset>/
```

For example:

```text
data/NielsenHB_2014/gene_families_abundance.csv
data/NielsenHB_2014/pathway_abundance_abundance.csv
data/NielsenHB_2014/relative_abundance_abundance.csv
```

The preprocessing notebooks use `data_process/prepare_mgp_config.py` to locate input files. You can switch datasets with environment variables before running the notebooks:

```bash
export MGP_DATA_ROOT=/path/to/MGPAN/data
export MGP_HUMANN_DATASET=NielsenHB_2014
export MGP_EDGE_DATASET=NielsenHB_2014
```

On Windows CMD:

```bat
set MGP_DATA_ROOT=D:\path\to\MGPAN\data
set MGP_HUMANN_DATASET=NielsenHB_2014
set MGP_EDGE_DATASET=NielsenHB_2014
```

预处理 notebook 会通过 `data_process/prepare_mgp_config.py` 读取输入路径。运行 notebook 前，可以用环境变量切换数据集。完整预处理细节见：

```text
data_process/README.md
```

## Reproduction Workflow / 复现流程

Run the following commands from the repository root.

在仓库根目录下依次执行以下步骤。

### 1. Prepare HUMAnN Input / 准备 HUMAnN 输入

```bash
jupyter notebook data_process/01_prepare_mgp_inputs_and_edges.ipynb
```

### 2. Build Graph Data / 构建图数据

```bash
jupyter notebook data_process/02_build_graph_dataset.ipynb
```

### 3. Train and Evaluate / 训练与评估

```bash
python main.py
```

## Central Configuration / 统一配置

Training, model, path, and evaluation defaults are centralized in `config.py`. For a standard run, edit `MGPANConfig` first and then run `python main.py`.

训练参数、模型参数、路径参数和评估参数都已集中到 `config.py`。标准复现时，建议先修改 `MGPANConfig`，再运行 `python main.py`。

Current key defaults:

```python
@dataclass
class MGPANConfig:
    dataset: str = "NielsenHB_2014"
    graphdata: str = "NH_graphdataF.bin"
    metadata: str = "NH_graphdata_metaF.pkl"
    subject_ids: str = "subject_ids.csv"

    data_dir: str = "./data"
    metapath_dir: str = "./metapaths"
    saved_model_dir: str = "saved_models"
    log_dir: str = "logs"

    metapath: str = "F_metapath_graphs8(full)"
    experimental: str = "F1metapath8+typepooling1+graphaugment+pos_weight(4+64+64)+1cl"
    log: str = "log_Mp_MGPAN1"
    model_name: str = "MGPAN"

    seed: int = 66
    device: str = "auto"
    n_splits: int = 10

    gnn: str = "sage"
    num_gnn_layer: int = 1
    embed_dim: int = 192
    dim_a: int = 56
    sage_aggregator: str = "pool"

    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.004
    contrastive_weight: float = 0.5
```

Command-line arguments can still override `config.py` values for a single run:

也可以用命令行参数临时覆盖 `config.py` 中的默认值：

```bash
python main.py \
  --dataset NielsenHB_2014 \
  --graphdata NH_graphdataF.bin \
  --metadata NH_graphdata_metaF.pkl \
  --epochs 100 \
  --batch-size 32
```

Some useful overrides:

```bash
python main.py --device cuda
python main.py --lr 0.001 --weight-decay 0.004
python main.py --edge_drop_prob 0.05 --node_drop_prob 0.0 --feat_mask_prob 0.0
python main.py --auto-pos-class-weight false --pos-class-weight 1.0
```

## Output Paths / 输出路径

Output locations are controlled by `config.py`:

输出路径由 `config.py` 控制：

```text
metapath cache / Meta-path 图缓存:
  <metapath_dir>/<dataset>/<metapath>/

logs / 日志:
  <log_dir>/<dataset>/metapaths/<log>.out

models / 模型:
  <saved_model_dir>/<dataset>/metapaths/<log>/

figures and raw predictions / 图像和原始预测:
  <data_dir>/<dataset>/figures_<log>/
```

With the current defaults, examples are:

按当前默认配置，示例如下：

```text
metapaths/NielsenHB_2014/F_metapath_graphs8(full)/
logs/NielsenHB_2014/metapaths/log_Mp_MGPAN1.out
saved_models/NielsenHB_2014/metapaths/log_Mp_MGPAN1/
data/NielsenHB_2014/figures_log_Mp_MGPAN1/
```

## Notes / 说明

- `DEFAULT_METAPATHS` and `RELATIONS` are defined in `config.py`.
- `main.py` reads all runtime parameters from `config.py` through `parse_args()`.
- `model/mgpan.py` also uses `MGPANConfig` defaults for standalone calls.
- Command-line options override config values only for the current run.

- `DEFAULT_METAPATHS` 和 `RELATIONS` 定义在 `config.py` 中。
- `main.py` 通过 `parse_args()` 从 `config.py` 读取运行参数。
- `model/mgpan.py` 在单独调用时也会使用 `MGPANConfig` 中的默认值。
- 命令行参数只会临时覆盖本次运行，不会改写 `config.py` 文件。
