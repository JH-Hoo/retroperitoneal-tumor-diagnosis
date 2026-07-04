# Binary Screening Hyperparameter Exploration

更新时间：2026-07-04

## 目的

本轮围绕当前二分类主线继续探索：

> 良性神经源性肿瘤 vs 非良性/需处理腹膜后肿瘤

优先级按本轮讨论固定为：

1. crop 只做轻量验证；
2. 重点看阈值策略；
3. 重点看 slice 采样策略；
4. 重点看 CT 窗设置。

阳性类仍是“非良性/需处理病变”。筛查任务下最重要的错误是 false negative：非良性/需处理病变被判成良性神经源性肿瘤。

## TotalSeg/Crop 评估

之前的 TotalSeg 方案不是病灶分割，而是基于器官/解剖结构构造一个腹膜后区域大框。远端仍保留了 30 例 pilot 的 ROI JSON 和 QC 图。

对 30 例 ROI JSON 统计：

```text
volume_fraction min    0.425
volume_fraction median 0.610
volume_fraction max    0.765
```

这说明 TotalSeg 框通常保留原始 CT volume 的 40%-75%，中位数约 61%。它确实能排除一部分无关区域，但仍然非常大，不是 lesion crop。因此不适合作为主要改进方向。

轻量 body crop 的二分类 image-only 对照如下：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base96 whole image | 0.776 | 0.604 | 0.616 | 0.916 | 0.291 | 0.689 | 0.879 | `[[16,39],[16,175]]` |
| bodyxy96 crop | 0.764 | 0.576 | 0.584 | 0.916 | 0.236 | 0.686 | 0.887 | `[[13,42],[16,175]]` |

结论：body-level crop 没有提升，specificity 还下降。这个结果支持“TotalSeg/大框 crop 不值得作为主线”的判断。

## Slice 采样策略

本轮测试了几种 slice 策略：

| 设置 | 说明 |
|---|---|
| `base96` | 原始 96 张全 z 轴均匀采样 |
| `uniform64` | 从原始 96-slice feature 中均匀取 64 个 feature |
| `center64` | 从原始 96-slice feature 中取中间连续 64 个 feature |
| `uniform48` | 从原始 96-slice feature 中均匀取 48 个 feature |
| `body_z96` | 从原始 NIfTI 重新生成，只在 HU > -500 的 body z 范围内均匀取 96 张 |

统一训练设置：

```text
POOLING=meanmax_mlp
FUSION=0
SAMPLER=binary50_subtype50
SELECT_METRIC=auroc
THRESHOLD_MODE=sens90
EPOCHS=80
```

结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base96 | 0.776 | 0.604 | 0.616 | 0.916 | 0.291 | 0.689 | 0.879 | `[[16,39],[16,175]]` |
| uniform64 | 0.764 | 0.563 | 0.567 | 0.927 | 0.200 | 0.644 | 0.857 | `[[11,44],[14,177]]` |
| center64 | 0.756 | 0.610 | 0.618 | 0.874 | 0.345 | 0.687 | 0.870 | `[[19,36],[24,167]]` |
| uniform48 | 0.772 | 0.588 | 0.598 | 0.921 | 0.255 | 0.702 | 0.886 | `[[14,41],[15,176]]` |
| body_z96 | 0.744 | 0.596 | 0.602 | 0.864 | 0.327 | 0.608 | 0.847 | `[[18,37],[26,165]]` |

结论：

- `uniform48` 的 AUROC 最高，达到 0.702，但 high-sensitivity operating point 下 specificity 只有 0.255。
- `center64` 提高了 specificity，但 sensitivity 明显下降。
- `body_z96` 不理想，说明简单去掉 z 轴空气/边缘并不能替代病灶定位。
- 当前不建议把 slice 子集作为默认主线。

## CT 窗设置

原始三窗：

```text
[-160, 240]
[-200, 100]
[-200, 400]
```

本轮全量重跑了一个更偏脂肪/宽窗的方案：

```text
[-150, 250]
[-250, 50]
[-300, 500]
```

结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| base96 原始窗 | 0.776 | 0.604 | 0.616 | 0.916 | 0.291 | 0.689 | 0.879 | `[[16,39],[16,175]]` |
| fat/wide 窗 | 0.760 | 0.600 | 0.609 | 0.890 | 0.309 | 0.670 | 0.864 | `[[17,38],[21,170]]` |

结论：

- 更宽的 fat/wide 方案没有提升；
- false negative 从 16 增加到 21；
- 当前原始三窗仍应保留为默认方案。

## 融合实验

把新 slice/window 变体加入旧融合池后，验证集选权重仍没有超过旧最佳。

旧最佳：

```text
Accuracy 0.821
Balanced Acc 0.691
Macro-F1 0.711
Sensitivity 0.927
Specificity 0.455
AUROC 0.670
Confusion Matrix [[25,30],[14,177]]
```

新增 slice/window 融合结果：

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| old + slice variants, screening/sens90 | 0.780 | 0.651 | 0.661 | 0.885 | 0.418 | 0.701 | 0.884 | `[[23,32],[22,169]]` |
| old + slice variants, screening/sens85 | 0.776 | 0.668 | 0.672 | 0.864 | 0.473 | 0.701 | 0.884 | `[[26,29],[26,165]]` |
| old + window variant, screening/sens90 | 0.768 | 0.650 | 0.655 | 0.864 | 0.436 | 0.676 | 0.874 | `[[24,31],[26,165]]` |
| old + window variant, screening/sens85 | 0.764 | 0.673 | 0.669 | 0.838 | 0.509 | 0.676 | 0.874 | `[[28,27],[31,160]]` |

虽然 `old + slice variants` 的 AUROC 达到 0.701，但在筛查关注的 high-sensitivity 区间没有转化成更好的 FN/FP 平衡。

## 阈值策略

阈值扫描说明阈值对 FN/FP 影响很大。以下为 pooled test 上的探索性全局阈值扫描，用来理解 operating point，不替代验证集选阈值。

旧最佳模型概率的全局阈值扫描：

| 目标 sensitivity | 实际 sensitivity | Specificity | FN | FP | 混淆矩阵 |
|---:|---:|---:|---:|---:|---|
| 0.95 | 0.953 | 0.345 | 9 | 36 | `[[19,36],[9,182]]` |
| 0.93 | 0.932 | 0.400 | 13 | 33 | `[[22,33],[13,178]]` |
| 0.90 | 0.901 | 0.455 | 19 | 30 | `[[25,30],[19,172]]` |
| 0.85 | 0.853 | 0.491 | 28 | 28 | `[[27,28],[28,163]]` |
| 0.80 | 0.801 | 0.509 | 38 | 27 | `[[28,27],[38,153]]` |

已报告的旧最佳主结果使用每折 validation 选阈值，而不是 pooled test 全局阈值：

```text
[[25,30],[14,177]]
Sensitivity 0.927
Specificity 0.455
```

结论：

- 阈值策略比本轮 slice/window/crop 改动更影响最终 FN/FP；
- 当前主线应继续使用 validation-selected high-sensitivity threshold；
- 报告中建议同时给出 `sens90` 和 `sens85` 两个 operating point，而不是只报 accuracy。

## 本轮结论

1. TotalSeg/大 body crop 框太粗，30 例 pilot 的体积占比中位数约 0.61，不是 lesion crop；body crop 实测没有提升。
2. slice 子集策略没有超过 96-slice baseline。`uniform48` AUROC 略高，但 high-sensitivity specificity 太差。
3. `body_z96` 不理想，单纯去掉空气/边缘 z 范围不够。
4. 更脂肪/宽窗的 CT 窗组合没有提升，原三窗应继续作为默认。
5. 当前最好结果仍是旧最佳 high-sensitivity fusion：`[[25,30],[14,177]]`。

下一步如果继续提升，优先级建议：

1. 保持当前三窗和 96-slice baseline；
2. 继续把阈值策略作为主要报告对象；
3. 如果做 crop，只做真正 lesion-level 的低成本粗框/中心点，不再投入 TotalSeg 大框；
4. 若继续窗宽窗位实验，应少量、有医学假设地做，而不是大规模随机网格。
