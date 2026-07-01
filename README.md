# 腹膜后肿瘤诊断项目

这个目录是当前项目的唯一主目录。训练、数据集、源表和实验输出都从这里读取或写入。

## 目录

- `dataset_standard_v0/`：标准母版数据集，41 个 CT NIfTI、标签、split、metadata 和 checksum。
- `dataset_96slice_v0/`：离线预处理缓存数据集，每例 96 张三窗轴位 slice，用于加速 MIL 训练。
- `dataset_96slice_balanced_aug_v0/`：尝试版类别均衡扩增缓存，只扩训练集，不扩测试集。
- `code/`：训练代码。
- `experiments/`：模型训练输出。
- `source_tables/`：原始 Excel / 病理表副本，用于追溯标签和影像筛选来源。
- `metadata/`：最终整理出的 41 例可用性和病理标签表。
- `audit/`：迁移和数据筛选过程中的审计记录。

## 当前数据

- 图像：41 个静脉期/门脉期薄层 CT NIfTI
- 训练集：37 例
- 测试集：4 例，每个四分类标签 1 例
- 四分类：肉瘤类、良性神经源性肿瘤、副神经节瘤、淋巴瘤

## 训练入口

```bash
python3 /Volumes/My_Drive/腹膜后肿瘤诊断/code/train_mil_demo.py
```

脚本会读取 `dataset_standard_v0/all.csv`，输出到 `experiments/`。

缓存版训练入口：

```bash
python3 /Volumes/My_Drive/腹膜后肿瘤诊断/code/train_mil_cached_demo.py
```

类别均衡扩增尝试版训练入口：

```bash
python3 /Volumes/My_Drive/腹膜后肿瘤诊断/code/train_mil_balanced_aug_demo.py
```
