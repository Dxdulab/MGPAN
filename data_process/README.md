# Data Preprocessing Guide / 数据预处理教程

## English

This folder contains the preprocessing notebooks used to prepare MGPAN graph inputs.

The first notebook, `01_prepare_mgp_inputs_and_edges.ipynb`, expects a HUMAnN merged file named:

```text
path_taxonomy_uf90.tsv
```

This file can be generated with HUMAnN's `merge_abundance.py` script from the official biobakery HUMAnN repository:

```text
https://github.com/biobakery/humann/blob/master/humann/tools/merge_abundance.py
```

Do not copy a single `merge_abundance.py` file into this project. The script imports other HUMAnN modules and default MetaCyc mapping files, so the reproducible approach is to clone and install the full HUMAnN repository.

## 中文

该目录包含用于准备 MGPAN 图输入的预处理 notebook。

第一个 notebook `01_prepare_mgp_inputs_and_edges.ipynb` 需要一个 HUMAnN 合并后的文件：

```text
path_taxonomy_uf90.tsv
```

这个文件可以通过 biobakery 官方 HUMAnN 仓库中的 `merge_abundance.py` 生成：

```text
https://github.com/biobakery/humann/blob/master/humann/tools/merge_abundance.py
```

不建议只复制单个 `merge_abundance.py` 到本项目中运行。该脚本会依赖 HUMAnN 内部模块和默认 MetaCyc mapping 文件，因此更可复现的做法是 clone 并安装完整 HUMAnN 仓库。

## 1. Clone HUMAnN / 克隆 HUMAnN

### English

From any working directory outside this repository:

```bash
git clone https://github.com/biobakery/humann.git external/humann
```

If you already cloned HUMAnN elsewhere, use that path instead.

### 中文

在本项目之外的任意工作目录执行：

```bash
git clone https://github.com/biobakery/humann.git external/humann
```

如果你已经在其他位置下载过 HUMAnN，可以直接使用已有路径。

## 2. Install HUMAnN / 安装 HUMAnN

### English

Activate the environment you use for preprocessing:

```bash
conda activate mgpan_preprocess
```

Then install HUMAnN from source:

```bash
pip install humann==3.9
```




### 中文

先激活用于预处理的环境：

```bash
conda activate mgpan_preprocess
```

然后以源码 editable 方式安装 HUMAnN：

```bash
pip install humann==3.9
```



## 3. Prepare Required TSV Files / 准备 TSV 输入文件

### English

For each dataset, place the abundance files under:

```text
MGPAN/data/<dataset>/
```

Required inputs before running `merge_abundance.py`:

```text
gene_families_abundance.tsv
pathway_abundance_abundance.tsv
```

If your files are still CSV files, open `01_prepare_mgp_inputs_and_edges.ipynb` and run the first configuration cell plus the `Convert CSV to TSV` cell. The notebook writes:

```text
gene_families_abundance.tsv
pathway_abundance_abundance.tsv
```

### 中文

每个数据集的丰度文件应放在：

```text
MGPAN/data/<dataset>/
```

运行 `merge_abundance.py` 前需要准备：

```text
gene_families_abundance.tsv
pathway_abundance_abundance.tsv
```

如果你的文件目前仍然是 CSV 格式，请打开 `01_prepare_mgp_inputs_and_edges.ipynb`，先运行第一格配置代码和 `Convert CSV to TSV` 这一格。notebook 会生成：

```text
gene_families_abundance.tsv
pathway_abundance_abundance.tsv
```

## 4. Run merge_abundance.py / 运行 merge_abundance.py

### English

Linux/macOS example:

```bash
cd /path/to/MGPAN/data/QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

Windows CMD example:

```bat
cd /d D:\path\to\MGPAN\data\QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

Windows PowerShell example:

```powershell
cd D:\path\to\MGPAN\data\QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

Expected output:

```text
MGPAN/data/QinN_2014/path_taxonomy_uf90.tsv
```

### 中文

Linux/macOS 示例：

```bash
cd /path/to/MGPAN/data/QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

Windows CMD 示例：

```bat
cd /d D:\path\to\MGPAN\data\QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

Windows PowerShell 示例：

```powershell
cd D:\path\to\MGPAN\data\QinN_2014
python -m python external\humann\tools\merge_abundance.py --input-genes gene_families_abundance.tsv --input-pathways pathway_abundance_abundance.tsv -o path_taxonomy_uf90.tsv
```

期望输出文件：

```text
MGPAN/data/QinN_2014/path_taxonomy_uf90.tsv
```

## 5. Verify the Output / 检查输出

### English

Run this from the dataset directory:

```bash
python -c "from pathlib import Path; p=Path('path_taxonomy_uf90.tsv'); assert p.exists() and p.stat().st_size > 0; print(p.resolve())"
```

Then continue running the remaining cells in:

```text
data_process/01_prepare_mgp_inputs_and_edges.ipynb
```

The notebook configuration file is:

```text
data_process/prepare_mgp_config.py
```

By default it reads data from `MGPAN/data/<dataset>/`. You can override paths with:

```bash
export MGP_DATA_ROOT=/path/to/MGPAN/data
export MGP_HUMANN_DATASET=QinN_2014
export MGP_EDGE_DATASET=QinN_2014
```

Windows CMD:

```bat
set MGP_DATA_ROOT=D:\path\to\MGPAN\data
set MGP_HUMANN_DATASET=QinN_2014
set MGP_EDGE_DATASET=QinN_2014
```

### 中文

在数据集目录下运行：

```bash
python -c "from pathlib import Path; p=Path('path_taxonomy_uf90.tsv'); assert p.exists() and p.stat().st_size > 0; print(p.resolve())"
```

确认文件存在且非空后，继续运行：

```text
data_process/01_prepare_mgp_inputs_and_edges.ipynb
```

notebook 的路径配置文件是：

```text
data_process/prepare_mgp_config.py
```

默认会从 `MGPAN/data/<dataset>/` 读取数据。你也可以通过环境变量覆盖路径：

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

## Troubleshooting / 常见问题

### English

If Python cannot import `humann`, make sure `pip install -e external/humann` was run in the same conda environment used by Jupyter.

If HUMAnN reports missing mapping files, reinstall the full HUMAnN package from source instead of running a copied standalone `merge_abundance.py` file.

If `path_taxonomy_uf90.tsv` is empty, check that the gene and pathway TSV files were generated from matched HUMAnN abundance outputs from the same dataset.

### 中文

如果 Python 无法导入 `humann`，请确认 `pip install -e external/humann` 是在 Jupyter 使用的同一个 conda 环境中执行的。

如果 HUMAnN 报告缺少 mapping 文件，请重新安装完整 HUMAnN 源码包，而不是运行单独复制出来的 `merge_abundance.py`。

如果 `path_taxonomy_uf90.tsv` 为空，请检查 gene 和 pathway TSV 是否来自同一个数据集、同一套 HUMAnN 输出。
