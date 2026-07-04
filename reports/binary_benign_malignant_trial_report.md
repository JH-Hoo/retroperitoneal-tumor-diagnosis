# 二分类非良性筛查报告

更新时间：2026-07-04

## 任务定义

当前项目主线固定为：

> 良性神经源性肿瘤 vs 非良性/需处理腹膜后肿瘤

这不是严格病理学意义上的“良恶性分类”。PPGL 不适合简单称为恶性，但它也不应该和普通良性神经源性肿瘤放在同一类，所以本项目把 PPGL 归入“非良性/需处理病变”。

| 二分类标签 | 纳入原 5 类 | 例数 |
|---|---|---:|
| 良性神经源性肿瘤 | 良性神经源性肿瘤 | 55 |
| 非良性/需处理病变 | 肉瘤类、淋巴瘤、PPGL、胃肠道间质瘤 | 191 |

总计 246 例，使用患者级 5-fold split。每一折都有独立 train、validation、test；最后报告的 pooled test 指标是把 5 个 fold 的 test 预测合并，所以混淆矩阵总数是 246，而不是单折几十例。

## 输入与预处理

输入来自全图 96-slice ResNet18 特征缓存：

- 每例 CT 均匀采样 96 张轴位 slice；
- 每张 slice resize 到 `224 x 224`；
- 三通道 CT 窗：
  - soft tissue：`[-160, 240]`
  - fat-sensitive：`[-200, 100]`
  - wide abdomen：`[-200, 400]`
- ImageNet 预训练 ResNet18 提取每张 slice 的 512 维特征；
- 训练阶段不使用 ROI、不使用分割、不使用肿瘤中心点；
- 表格变量只使用年龄和性别。

## 模型结构

训练入口为 `scripts/20_train_binary_feature_fusion.py`。

当前脚本支持几种轻量 MIL head：

| Head | 图像聚合方式 | 用途 |
|---|---|---|
| `meanmax` | 96 张 slice feature 做 mean pooling 和 max pooling 后拼接 | 最稳的基础影像模型 |
| `meanmax_mlp` | `meanmax` 后接 LayerNorm + MLP | 小幅增加非线性 |
| `topk8_zpos` | 学习 slice score，取 top-k slice，并加入 z-position embedding | 尝试让模型关注少数关键层面 |
| `gated_attention_zpos` | gated attention MIL，并加入 z-position embedding | 尝试学习 slice 权重 |
| `metadata_only` | 只使用年龄、性别 | 检查表格变量本身的贡献 |

训练还加入了：

- binary-balanced 或 subtype-balanced sampler；
- class-weighted CE、普通 CE、focal loss 可选；
- validation 阈值选择：
  - 固定 `0.5`；
  - Youden；
  - screening 阈值：在 validation sensitivity 达到 0.90 或 0.85 时尽量提高 specificity。

## 主要结果

下表为 5-fold pooled test 指标。阳性类是“非良性/需处理病变”，所以 sensitivity 表示非良性/需处理病例被识别出来的比例；false negative 是最需要警惕的错误。

| 设置 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | AP | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 旧主线：ResNet18 mean/max + 年龄/性别 | 0.789 | 0.650 | 0.664 | 0.901 | 0.400 | 0.698 | 0.863 | `[[22,33],[19,172]]` |
| 年龄/性别 only，sens85 阈值 | 0.805 | 0.661 | 0.679 | 0.921 | 0.400 | 0.671 | 0.869 | `[[22,33],[15,176]]` |
| 图像 only mean/max，固定 0.5 | 0.732 | 0.614 | 0.614 | 0.827 | 0.400 | 0.666 | 0.869 | `[[22,33],[33,158]]` |
| 图像 only gated attention，固定 0.5 | 0.667 | 0.656 | 0.610 | 0.675 | 0.636 | 0.643 | 0.850 | `[[35,20],[62,129]]` |
| 推荐筛查点：metadata + gated MIL fusion，sens85 阈值 | 0.813 | 0.679 | 0.698 | 0.921 | 0.436 | 0.642 | 0.833 | `[[24,31],[15,176]]` |
| 推荐平衡点：metadata + image-only mean/max late fusion，sens85 阈值 | 0.805 | 0.680 | 0.694 | 0.906 | 0.455 | 0.712 | 0.871 | `[[25,30],[18,173]]` |
| 更高 specificity 点：metadata + baseline fusion，固定 0.5 | 0.785 | 0.686 | 0.688 | 0.864 | 0.509 | 0.698 | 0.863 | `[[28,27],[26,165]]` |

![二分类指标对比](assets/binary_trial_metrics.png)

![二分类混淆矩阵](assets/binary_trial_confusion.png)

## 推荐结果怎么理解

如果目标是筛查，最担心的是“非良性/需处理病变被判成良性神经源性肿瘤”，也就是 false negative。按这个目标，当前更适合作为主结果的是：

```text
metadata + gated MIL fusion，validation sens85 threshold
Accuracy 0.813
Balanced Accuracy 0.679
Macro-F1 0.698
Sensitivity 0.921
Specificity 0.436
Confusion Matrix [[24,31],[15,176]]
```

它相对旧主线的变化是：

| 指标 | 旧主线 | 当前推荐筛查点 |
|---|---:|---:|
| False negative | 19 | 15 |
| False positive | 33 | 31 |
| Accuracy | 0.789 | 0.813 |
| Balanced Acc | 0.650 | 0.679 |
| Macro-F1 | 0.664 | 0.698 |
| Sensitivity | 0.901 | 0.921 |
| Specificity | 0.400 | 0.436 |

也就是说，它没有只靠“更激进地全判阳性”来提高 sensitivity，而是在 false negative 和 false positive 上都比旧主线略好。

如果更重视 AUROC 和影像贡献的可解释性，可以把 `metadata + image-only mean/max late fusion` 作为备选主结果。它的 sensitivity 为 0.906，specificity 为 0.455，AUROC 为 0.712，说明这个 late fusion 比单纯年龄/性别更像一个可继续发展的影像模型。

## 亚型表现

推荐筛查点 `metadata + gated MIL fusion` 的 pooled subtype recall：

| 原 5 类 | n | 二分类目标 | Recall |
|---|---:|---:|---:|
| 肉瘤类 | 103 | 非良性/需处理 | 0.942 |
| 良性神经源性肿瘤 | 55 | 良性神经源性肿瘤 | 0.436 |
| PPGL | 30 | 非良性/需处理 | 0.867 |
| 淋巴瘤 | 44 | 非良性/需处理 | 0.909 |
| 胃肠道间质瘤 | 14 | 非良性/需处理 | 0.929 |

主要短板仍是良性神经源性肿瘤的 specificity/recall 不高，55 例里只有 24 例被正确识别为良性，31 例被模型判为非良性/需处理。

## 结论

这轮优化支持三个判断：

1. 二分类方向比五分类更适合当前数据。
2. 年龄/性别本身已经很强，必须作为对照报告，不能把所有提升都归因于 CT 影像。
3. 在不做 ROI、不做中心点、不做分割的前提下，轻量 late fusion 可以把旧主线从 `[[22,33],[19,172]]` 推到 `[[24,31],[15,176]]` 或 `[[25,30],[18,173]]`，属于有价值但仍有限的提升。

下一步如果要继续提高 specificity，最可能有效的不是再堆复杂 head，而是引入极低成本的人工肿瘤中心点、粗框或 lesion crop，让模型少看无关背景。
