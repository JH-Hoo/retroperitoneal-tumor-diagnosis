# Binary Slice And Window Sweep

更新时间：2026-07-05

## 目的

本轮继续扫二分类主线的两个输入超参数：

1. slice 采样策略，重点加入 Gaussian 分布采样；
2. CT 三窗设置，继续测试脂肪窗、软组织窗和宽窗组合。

任务仍是：

> 良性神经源性肿瘤 vs 非良性/需处理腹膜后肿瘤

阳性类是“非良性/需处理病变”。筛查场景下首要关注 false negative。

## 实验设置

统一训练设置：

```text
POOLING=meanmax_mlp
FUSION=0
SAMPLER=binary50_subtype50
LOSS=ce
SELECT_METRIC=auroc
THRESHOLD_MODE=sens90
EPOCHS=80
BATCH_SIZE=16
LR=0.001
SCHEDULER=cosine
CLIP_NORM=5
```

对所有设置均做 5-fold patient-level CV，并报告 pooled test。

## Gaussian Slice Sampling

Gaussian 采样从已有 96-slice ResNet18 feature 中抽取子集。参数：

```text
GAUSS_MU    = 采样中心，0.5 为中间层面，0.4/0.6 为偏上/偏下
GAUSS_SIGMA = 集中程度，越小越集中
```

本轮测试：

| 设置 | n slices | mu | sigma | 说明 |
|---|---:|---:|---:|---|
| `gauss64_mid_s18` | 64 | 0.50 | 0.18 | 中心更集中 |
| `gauss64_mid_s26` | 64 | 0.50 | 0.26 | 中心较宽 |
| `gauss48_mid_s22` | 48 | 0.50 | 0.22 | 更少 slice |
| `gauss64_mu40_s22` | 64 | 0.40 | 0.22 | 偏上/前半 z 范围 |
| `gauss64_mu60_s22` | 64 | 0.60 | 0.22 | 偏下/后半 z 范围 |

结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `gauss64_mid_s18` | 0.776 | 0.604 | 0.616 | 0.916 | 0.291 | 0.678 | 0.874 | `[[16,39],[16,175]]` |
| `gauss64_mid_s26` | 0.768 | 0.611 | 0.622 | 0.895 | 0.327 | 0.682 | 0.874 | `[[18,37],[20,171]]` |
| `gauss48_mid_s22` | 0.785 | 0.615 | 0.630 | 0.921 | 0.309 | 0.675 | 0.865 | `[[17,38],[15,176]]` |
| `gauss64_mu40_s22` | 0.785 | 0.628 | 0.643 | 0.911 | 0.345 | 0.693 | 0.871 | `[[19,36],[17,174]]` |
| `gauss64_mu60_s22` | 0.772 | 0.607 | 0.619 | 0.906 | 0.309 | 0.691 | 0.877 | `[[17,38],[18,173]]` |

结论：

- 最好的是 `gauss64_mu40_s22`，但仍未超过旧最佳。
- Gaussian 采样略能提高 AUROC 或 BAcc，但 high-sensitivity specificity 仍偏低。
- 偏上 `mu40` 比偏下 `mu60` 稍好，但差异不足以改主线。

## CT Window Sweep

对比基线三窗：

```text
baseline:
[-160, 240]
[-200, 100]
[-200, 400]
```

新增三组窗口：

| 设置 | 三窗 |
|---|---|
| `win_fat_narrow96` | `[-160,240] / [-250,0] / [-300,500]` |
| `win_soft_fat96` | `[-100,300] / [-250,50] / [-200,500]` |
| `win_ultrawide96` | `[-200,400] / [-300,100] / [-500,800]` |

结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `win_fat_narrow96` | 0.768 | 0.585 | 0.595 | 0.916 | 0.255 | 0.676 | 0.869 | `[[14,41],[16,175]]` |
| `win_soft_fat96` | 0.752 | 0.581 | 0.589 | 0.890 | 0.273 | 0.641 | 0.852 | `[[15,40],[21,170]]` |
| `win_ultrawide96` | 0.752 | 0.601 | 0.608 | 0.874 | 0.327 | 0.699 | 0.882 | `[[18,37],[24,167]]` |

结论：

- `win_ultrawide96` 的 AUROC 最高，达到 0.699；
- 但它的 sensitivity 只有 0.874，false negative 为 24；
- 更窄脂肪窗没有带来明显收益；
- `win_soft_fat96` 整体最弱；
- 当前不建议替换基线三窗。

## Fusion Check

把较好的 Gaussian/window 候选加入旧融合池：

```text
metadata-only
old image-only
old gated MIL
uniform48
gauss64_mu40_s22
gauss64_mu60_s22
win_ultrawide96
win_fat_narrow96
```

结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fusion, AUROC select, sens90 | 0.744 | 0.641 | 0.638 | 0.827 | 0.455 | 0.657 | 0.860 | `[[25,30],[33,158]]` |
| fusion, screening select, sens90 | 0.740 | 0.625 | 0.625 | 0.832 | 0.418 | 0.699 | 0.876 | `[[23,32],[32,159]]` |
| fusion, screening select, sens85 | 0.732 | 0.620 | 0.619 | 0.822 | 0.418 | 0.699 | 0.876 | `[[23,32],[34,157]]` |

旧最佳仍为：

```text
Accuracy 0.821
Balanced Acc 0.691
Macro-F1 0.711
Sensitivity 0.927
Specificity 0.455
AUROC 0.670
Confusion Matrix [[25,30],[14,177]]
```

结论：新增候选提供了一些排序信号，但加入融合池后没有改善 high-sensitivity operating point，反而增加 false negative。

## Threshold Operating Points

几个代表模型的全局阈值扫描如下。该扫描用于理解阈值行为；正式结果仍以 validation-selected threshold 为主。

### 旧最佳

| target sensitivity | sensitivity | specificity | FN | FP | 混淆矩阵 |
|---:|---:|---:|---:|---:|---|
| 0.95 | 0.953 | 0.345 | 9 | 36 | `[[19,36],[9,182]]` |
| 0.93 | 0.932 | 0.400 | 13 | 33 | `[[22,33],[13,178]]` |
| 0.90 | 0.901 | 0.455 | 19 | 30 | `[[25,30],[19,172]]` |
| 0.85 | 0.853 | 0.491 | 28 | 28 | `[[27,28],[28,163]]` |

### `gauss64_mu40_s22`

| target sensitivity | sensitivity | specificity | FN | FP | 混淆矩阵 |
|---:|---:|---:|---:|---:|---|
| 0.95 | 0.953 | 0.236 | 9 | 42 | `[[13,42],[9,182]]` |
| 0.93 | 0.932 | 0.255 | 13 | 41 | `[[14,41],[13,178]]` |
| 0.90 | 0.901 | 0.327 | 19 | 37 | `[[18,37],[19,172]]` |
| 0.85 | 0.853 | 0.364 | 28 | 35 | `[[20,35],[28,163]]` |

### `win_ultrawide96`

| target sensitivity | sensitivity | specificity | FN | FP | 混淆矩阵 |
|---:|---:|---:|---:|---:|---|
| 0.95 | 0.953 | 0.127 | 9 | 48 | `[[7,48],[9,182]]` |
| 0.93 | 0.932 | 0.200 | 13 | 44 | `[[11,44],[13,178]]` |
| 0.90 | 0.901 | 0.327 | 19 | 37 | `[[18,37],[19,172]]` |
| 0.85 | 0.853 | 0.382 | 28 | 34 | `[[21,34],[28,163]]` |

结论：

- 新候选在 AUROC 上可以接近或略高于旧最佳；
- 但在 0.90 以上 sensitivity 区间，specificity 都不如旧最佳；
- 因此不能只按 AUROC 选择模型。

## 本轮结论

1. Gaussian slice 采样可以作为候选特征保留，但不应替换 96-slice baseline。
2. Gaussian 中 `mu40/sigma0.22` 最好，但 high-sensitivity specificity 仍不足。
3. CT 窗 sweep 中 `ultrawide` AUROC 最高，但 FN 太多，不适合作为筛查主结果。
4. 更窄脂肪窗没有带来预期提升。
5. 旧最佳 high-sensitivity fusion 仍是当前主结果。

当前建议：

```text
主模型继续用旧最佳 high-sensitivity fusion。
输入默认仍用原三窗和 96-slice baseline。
报告可以补充说明：Gaussian/window sweep 未改善 high-sensitivity operating point。
下一步真正值得做的是 lesion-level 粗框或中心点，而不是继续大规模扫全图 slice/window。
```
