# Multiview Augmentation Trial

更新时间：2026-07-04

## 目的

本次试验验证真实数据增广是否能改善当前二分类主线：

> 良性神经源性肿瘤 vs 非良性/需处理腹膜后肿瘤

阳性类为“非良性/需处理病变”。筛查场景下最关注 false negative，也就是非良性/需处理病变被判成良性神经源性肿瘤。

## 实际执行内容

这次不是只写增广代码，已经实际生成 cache、提取特征并完成 5-fold 训练。

### Tensor Cache

目录：

```text
data/cache_96slice_aug5/
```

远端实际保存位置通过 symlink 指向：

```text
/root/retro_aug_cache/cache_96slice_aug5/
```

生成结果：

| 项目 | 数值 |
|---|---:|
| 病例数 | 249 |
| views per case | 5 |
| tensor 总数 | 1245 |
| tensor shape | `[96,3,224,224]` |
| dtype | `uint8` |

增广定义：

| view | 内容 |
|---|---|
| view0 | 固定均匀 96 slice，固定三窗 |
| view1 | z-jitter + window jitter |
| view2 | z-jitter + mild affine |
| view3 | z-jitter + window jitter + mild Gaussian noise |
| view4 | z-jitter + window jitter |

三窗仍为：

```text
soft tissue:   [-160, 240]
fat-sensitive: [-200, 100]
wide abdomen:  [-200, 400]
```

### Feature Cache

目录：

```text
data/features_cache_96slice_aug5_resnet18/
```

生成结果：

| 项目 | 数值 |
|---|---:|
| 可训练病例数 | 246 |
| views per case | 5 |
| feature tensor 总数 | 1230 |
| 每个 view shape | `[96,512]` |
| dtype | `float16` |
| backbone | ImageNet-pretrained ResNet18 |

训练时随机取一个 view；验证和测试时对 5 个 view 做概率平均。

## 训练设置

两组 image-only 模型：

| 运行名 | MIL head | 输入 |
|---|---|---|
| `p2_aug_meanmax_img_fold{fold}_auc_sens90` | `meanmax_mlp` | aug5 ResNet18 features |
| `p2_aug_gated_img_fold{fold}_auc_sens90` | `gated_attention_zpos` | aug5 ResNet18 features |

共同设置：

```text
5-fold patient-level CV
NUM_VIEWS=5
TRAIN_VIEW_MODE=random
TEST_VIEW_MODE=mean
SAMPLER=binary50_subtype50
SELECT_METRIC=auroc
THRESHOLD_MODE=sens90
EPOCHS=80
BATCH_SIZE=16
LR=0.001
SCHEDULER=cosine
CLIP_NORM=5
```

随后做 late fusion：

```text
metadata-only + aug meanmax + aug gated
```

并做额外增量检验：

```text
旧 metadata-only + 旧 image-only + 旧 gated + aug meanmax + aug gated
```

## 结果

5-fold pooled test，n=246。

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| aug meanmax image-only | 0.748 | 0.559 | 0.563 | 0.901 | 0.218 | 0.684 | 0.876 | `[[12,43],[19,172]]` |
| aug gated image-only | 0.744 | 0.550 | 0.552 | 0.901 | 0.200 | 0.638 | 0.858 | `[[11,44],[19,172]]` |
| metadata + aug meanmax + aug gated, sens90 | 0.772 | 0.627 | 0.638 | 0.890 | 0.364 | 0.667 | 0.859 | `[[20,35],[21,170]]` |
| metadata + aug meanmax + aug gated, sens85 | 0.768 | 0.657 | 0.660 | 0.859 | 0.455 | 0.667 | 0.859 | `[[25,30],[27,164]]` |
| 旧最佳 equal fusion, sens90 | 0.821 | 0.691 | 0.711 | 0.927 | 0.455 | 0.670 | 0.848 | `[[25,30],[14,177]]` |
| 旧最佳 equal fusion, sens85 | 0.793 | 0.692 | 0.695 | 0.874 | 0.509 | 0.670 | 0.848 | `[[28,27],[24,167]]` |

## 增量融合检验

把 aug 模型加入旧模型池后，验证集选权重仍没有超过旧最佳。

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| old + aug, AUROC select, sens90 | 0.744 | 0.634 | 0.634 | 0.832 | 0.436 | 0.660 | 0.852 | `[[24,31],[32,159]]` |
| old + aug, rank-score select, sens90 | 0.752 | 0.633 | 0.636 | 0.848 | 0.418 | 0.664 | 0.868 | `[[23,32],[29,162]]` |
| old + aug, screening select, sens90 | 0.772 | 0.653 | 0.659 | 0.869 | 0.436 | 0.664 | 0.858 | `[[24,31],[25,166]]` |

## 结论

这次真实 multiview augmentation 已经执行完，但没有改善当前主线。

最关键的比较是：

```text
旧最佳 sens90: [[25,30],[14,177]]
aug 融合 sens90: [[20,35],[21,170]]
```

也就是说，增广融合后 false negative 从 14 增加到 21，specificity 也没有提高。按筛查任务目标，这是退步。

单看 image-only，aug meanmax 的 AUROC 为 0.684，略高于某些旧融合结果的 AUROC，但在高敏感度阈值下 specificity 只有 0.218，说明排序信息有限，阈值落地效果不好。

当前建议：

1. 不把 aug5 作为新的默认主线。
2. 继续保留代码能力，因为正式数据扩大后仍可复用。
3. 当前报告主结果仍应采用旧最佳 high-sensitivity fusion：`Accuracy 0.821`、`Balanced Acc 0.691`、`Sensitivity 0.927`、`Specificity 0.455`、混淆矩阵 `[[25,30],[14,177]]`。
4. 下一步如果继续提升，优先考虑低成本 lesion crop/粗框，而不是继续堆随机 view 增广。
