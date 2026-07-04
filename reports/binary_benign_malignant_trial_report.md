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

总计 246 例，使用患者级 5-fold split。

## 模型与输入

训练入口为 `scripts/20_train_binary_feature_fusion.py`。

输入来自全图 96-slice ResNet18 特征缓存：

- 每例 CT 均匀采样 96 张轴位 slice；
- 每张 slice resize 到 `224 x 224`；
- 三通道 CT 窗：
  - soft tissue：`[-160, 240]`
  - fat-sensitive：`[-200, 100]`
  - wide abdomen：`[-200, 400]`
- ImageNet 预训练 ResNet18 提取每张 slice 的 512 维特征；
- 96 张 slice 特征做 mean/max pooling；
- 可选拼接年龄和性别；
- 使用 class-weighted cross entropy。

当前推荐设置是：

```bash
FUSION=1 FEATURE_NAME=features_cache_96slice_resnet18 POOLING=meanmax python scripts/20_train_binary_feature_fusion.py
```

## 结果

5 折 test set 合并后的 pooled 指标如下：

| 模型 | Accuracy | Balanced Acc | Macro-F1 | Sensitivity | Specificity | AUROC | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---:|---|
| 全图 ResNet18 特征 | 0.732 | 0.614 | 0.614 | 0.827 | 0.400 | 0.666 | `[[22,33],[33,158]]` |
| 全图 ResNet18 特征 + 年龄/性别 | 0.789 | 0.650 | 0.664 | 0.901 | 0.400 | 0.698 | `[[22,33],[19,172]]` |

![二分类指标对比](assets/binary_trial_metrics.png)

![二分类混淆矩阵](assets/binary_trial_confusion.png)

## 解释

这个结果说明二分类任务比原来的多分类任务更可行。模型可以比较积极地抓住“非良性/需处理”病例，融合年龄/性别后 sensitivity 达到 0.901。

主要问题是 specificity 偏低，良性神经源性肿瘤仍有较多被判为非良性/需处理。因此当前模型更适合作为探索性筛查 baseline，不适合表述为成熟的临床诊断模型。

年龄/性别融合在这个二分类任务中提高了 accuracy、macro-F1 和 sensitivity，但没有改善 specificity。它可以作为主线设置保留，同时在讨论中说明临床变量的贡献仍然有限。

## 结论

后续主线应围绕“良性神经源性肿瘤 vs 非良性/需处理病变”继续优化。最值得尝试的下一步是加入人工肿瘤中心点、粗框或 lesion crop，减少全图输入中的无关背景，提高良性病例的识别能力。
