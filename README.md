# 腹膜后肿瘤 CT 五分类

当前项目只保留 200+ CT 数据版本的主线：把静脉期/门脉期 NIfTI 预处理成 96-slice 三窗缓存，再训练无分割 2.5D attention MIL 五分类模型。

## 目录

```text
code/          训练、预处理、脱敏和 split 构建脚本
data/          可公开到私有 GitHub 的脱敏标签表、cache 索引和审计表
data_private/ 只在本机/远端保留的 PHI、salt、源表格和私有审计
models/        本地训练权重、预测和指标；权重不进 GitHub
envs/          CUDA / PyTorch 环境文件
reports/       实验报告和结果图
```

`data/` 内部：

```text
cache_96slice/                  249 例 96 x 3 x 224 x 224 tensor cache 的索引；tensor 实体不进 GitHub
labels_5class_groupcv_deid/     当前主线：脱敏 5-class patient-level 5-fold split
audits_deid/                    脱敏重复患者和旧 split 泄漏审计
```

## 当前数据

- 原始图像：252 个静脉期/门脉期 CT NIfTI
- 坏 gzip NIfTI：3 个，当前跳过 `G0122`、`G0137`、`G0369`
- 96-slice 缓存：249 例，tensor 文件不拉回本地、不进 GitHub
- 五分类监督样本：246 例
- 当前主线 split：5-fold `StratifiedGroupKFold`，按 salted `patient_uid_hash` 分组
- 五分类：肉瘤类、良性神经源性肿瘤、PPGL、淋巴瘤、胃肠道间质瘤
- 旧 holdout split 已发现 1 组同患者跨 train/test，只保留为早期 smoke test 结果解释，不再作为正式评价 split

## 主线脚本

构建 96-slice 缓存：

```bash
python code/build_96slice_dataset.py
```

生成脱敏 patient-level 5-fold 标签表：

```bash
python code/build_deid_group_splits.py
```

训练五分类 group-CV fold 0 模型：

```bash
python code/train_mil_cached_5class_holdout.py
```

## 模型

- 输入：每例 `96 x 3 x 224 x 224`
- 三个 CT 窗：`[-160,240]`、`[-200,100]`、`[-200,400]`
- backbone：ImageNet 预训练 ResNet18
- 聚合：attention MIL
- 输出：五分类 softmax
- 当前训练策略：先冻结 backbone，之后只解冻 `layer4`；BatchNorm 保持 eval
- 当前输出目录：`models/5class_groupcv_fold0/`

## GitHub 内容边界

GitHub 仓库只保存代码、环境文件、脱敏标签表、cache 索引、脱敏审计、训练日志、预测表、指标和报告。

不进入 GitHub：

- 原始 NIfTI
- 96-slice tensor cache
- `.pt/.pth` 模型权重
- 源 Excel
- 姓名、住院号、病理号、出生日期、病理原文、本地绝对路径
- hash salt 和带 PHI 的 linkage audit

## 环境

迁移到 RTX 3060/3080/3090 或 A5000 这类机器时，优先使用 CUDA 11.8 环境：

```bash
conda env create -f envs/environment-cu118.yml
conda activate rtp-mil-cu118
```
