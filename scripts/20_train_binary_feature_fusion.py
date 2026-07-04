#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

FEATURE_NAME = os.environ.get("FEATURE_NAME", "features_cache_96slice_resnet18")
POOLING = os.environ.get("POOLING", "meanmax")
FUSION = os.environ.get("FUSION", "0") == "1"
FOLD = int(os.environ.get("FOLD", "0"))
RUN_NAME = os.environ.get(
    "RUN_NAME",
    f"binary_nonbenign_{FEATURE_NAME}_fold{FOLD}_{POOLING}" + ("_age_sex_fusion" if FUSION else ""),
)
OUT_DIR = PROJECT_ROOT / "runs" / RUN_NAME
FEATURE_DIR = PROJECT_ROOT / "data" / FEATURE_NAME / "features"

EPOCHS = int(os.environ.get("EPOCHS", "80"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LR = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0001"))
DROPOUT = float(os.environ.get("DROPOUT", "0.2"))
SEED = int(os.environ.get("SEED", "20260704"))

SAMPLER = os.environ.get("SAMPLER", "natural")
LOSS = os.environ.get("LOSS", "weighted_ce")
FOCAL_GAMMA = float(os.environ.get("FOCAL_GAMMA", "2.0"))
SELECT_METRIC = os.environ.get("SELECT_METRIC", "macro_f1")
THRESHOLD_MODE = os.environ.get("THRESHOLD_MODE", "fixed_05")
MIN_SENSITIVITY = float(os.environ.get("MIN_SENSITIVITY", "0.90"))
FEATURE_NORM = os.environ.get("FEATURE_NORM", "1") == "1"
AUX_5CLASS = os.environ.get("AUX_5CLASS", "0") == "1"
AUX_WEIGHT = float(os.environ.get("AUX_WEIGHT", "0.3"))

CLASS_NAMES = ["benign_neurogenic", "nonbenign_actionable"]
LABEL5_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]


def read_rows(split):
    with (LABEL_DIR / "splits" / f"fold_{FOLD}" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def binary_target(row):
    label = row["label_5"]
    if label == "良性神经源性肿瘤":
        return 0
    if label in ["肉瘤类", "淋巴瘤", "PPGL", "胃肠道间质瘤"]:
        return 1
    raise ValueError(f"unhandled label for binary_nonbenign: {label}")


def label5_target(row):
    return int(row["label_5_id"])


class TabularEncoder:
    def __init__(self, rows):
        ages = np.asarray([float(r["age_at_scan"]) for r in rows], dtype=np.float32)
        self.age_mean = float(ages.mean())
        self.age_std = float(ages.std() if ages.std() > 0 else 1.0)
        self.sex_values = ["男", "女"]

    @property
    def dim(self):
        return 1 + len(self.sex_values)

    def encode(self, row):
        values = [(float(row["age_at_scan"]) - self.age_mean) / self.age_std]
        values.extend([float(row["sex"] == x) for x in self.sex_values])
        return torch.tensor(values, dtype=torch.float32)

    def state(self):
        return {"age_mean": self.age_mean, "age_std": self.age_std, "sex_values": self.sex_values}


class BinaryBags(Dataset):
    def __init__(self, rows, tabular_encoder):
        self.rows = rows
        self.tabular_encoder = tabular_encoder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        feat = torch.load(FEATURE_DIR / f"{row['case_id']}.pt", map_location="cpu").float()
        tab = self.tabular_encoder.encode(row)
        return (
            feat,
            tab,
            torch.tensor(binary_target(row), dtype=torch.long),
            torch.tensor(label5_target(row), dtype=torch.long),
            row["case_id"],
            row["label_5"],
        )


def topk_from_pooling():
    m = re.search(r"topk(\d+)", POOLING)
    return int(m.group(1)) if m else 8


def uses_zpos():
    return POOLING.endswith("_zpos") or "zpos" in POOLING


def make_classifier(dim, out_dim):
    if POOLING == "meanmax":
        return nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(dim, out_dim))
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Dropout(DROPOUT),
        nn.Linear(dim, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(256, out_dim),
    )


class BinaryMILHead(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.pooling = POOLING
        self.feature_norm = nn.LayerNorm(512) if FEATURE_NORM else nn.Identity()
        self.use_zpos = uses_zpos()
        if self.use_zpos:
            self.z_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True), nn.Linear(32, 512))

        self.force_tabular = POOLING == "metadata_only"
        if FUSION or self.force_tabular:
            self.tabular_branch = nn.Sequential(
                nn.Linear(tab_dim, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(16, 16),
                nn.ReLU(inplace=True),
            )
            tab_out = 16
        else:
            self.tabular_branch = None
            tab_out = 0

        if POOLING == "metadata_only":
            image_dim = 0
        elif POOLING in ["mean", "max"]:
            image_dim = 512
        elif POOLING in ["meanmax", "meanmax_mlp"]:
            image_dim = 1024
        elif POOLING.startswith("topk"):
            self.topk = topk_from_pooling()
            self.slice_scorer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
            self.feat_proj = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
            image_dim = 256
        elif POOLING.startswith("gated_attention"):
            self.V = nn.Linear(512, 128)
            self.U = nn.Linear(512, 128)
            self.attn_w = nn.Linear(128, 1)
            image_dim = 512
        elif POOLING.startswith("transformer"):
            layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            image_dim = 1024
        else:
            raise ValueError(f"unknown POOLING: {POOLING}")

        classifier_dim = image_dim + tab_out
        self.classifier = make_classifier(classifier_dim, 2)
        self.aux_classifier = make_classifier(classifier_dim, 5) if AUX_5CLASS else None

    def add_zpos(self, feat):
        if not self.use_zpos:
            return feat
        b, s, _ = feat.shape
        z = torch.linspace(0, 1, s, device=feat.device).view(1, s, 1).repeat(b, 1, 1)
        return feat + self.z_mlp(z)

    def image_pool(self, feat):
        feat = self.add_zpos(self.feature_norm(feat))
        if self.pooling == "metadata_only":
            weights = torch.zeros(feat.shape[:2], device=feat.device)
            return torch.empty((feat.shape[0], 0), device=feat.device), weights
        if self.pooling == "mean":
            weights = torch.full(feat.shape[:2], 1.0 / feat.shape[1], device=feat.device)
            return feat.mean(dim=1), weights
        if self.pooling == "max":
            weights = torch.zeros(feat.shape[:2], device=feat.device)
            return feat.max(dim=1).values, weights
        if self.pooling in ["meanmax", "meanmax_mlp"]:
            weights = torch.full(feat.shape[:2], 1.0 / feat.shape[1], device=feat.device)
            return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=1), weights
        if self.pooling.startswith("topk"):
            score = self.slice_scorer(feat).squeeze(-1)
            top_idx = score.topk(min(self.topk, feat.shape[1]), dim=1).indices
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            top_feat = feat.gather(dim=1, index=gather_idx)
            pooled = self.feat_proj(top_feat).mean(dim=1)
            return pooled, torch.softmax(score, dim=1)
        if self.pooling.startswith("gated_attention"):
            a = torch.tanh(self.V(feat)) * torch.sigmoid(self.U(feat))
            weights = torch.softmax(self.attn_w(a).squeeze(-1), dim=1)
            pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)
            return pooled, weights
        if self.pooling.startswith("transformer"):
            feat = self.encoder(feat)
            weights = torch.full(feat.shape[:2], 1.0 / feat.shape[1], device=feat.device)
            return torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=1), weights
        raise ValueError(self.pooling)

    def forward(self, feat, tab):
        image_feat, weights = self.image_pool(feat)
        if self.tabular_branch is not None:
            image_feat = torch.cat([image_feat, self.tabular_branch(tab)], dim=1)
        logits = self.classifier(image_feat)
        aux_logits = self.aux_classifier(image_feat) if self.aux_classifier is not None else None
        return logits, aux_logits, weights


def class_weights(rows):
    counts = np.bincount([binary_target(r) for r in rows], minlength=2)
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)


def class5_weights(rows):
    counts = np.bincount([label5_target(r) for r in rows], minlength=5)
    return torch.tensor(counts.sum() / (5 * counts), dtype=torch.float32)


def make_sampler(rows):
    if SAMPLER == "natural":
        return None
    if SAMPLER == "balanced":
        y = np.asarray([binary_target(r) for r in rows])
        counts = np.bincount(y, minlength=2)
        weights = (1.0 / counts)[y]
    elif SAMPLER == "subtype_balanced":
        labels = [r["label_5"] for r in rows]
        counts = {x: labels.count(x) for x in set(labels)}
        weights = np.asarray([1.0 / counts[x] for x in labels], dtype=np.float64)
    else:
        raise ValueError(f"unknown SAMPLER: {SAMPLER}")
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(rows), replacement=True)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_loss(logits, y, ce_weight):
    if LOSS == "ce":
        return F.cross_entropy(logits, y)
    if LOSS == "weighted_ce":
        return F.cross_entropy(logits, y, weight=ce_weight)
    if LOSS == "focal":
        ce = F.cross_entropy(logits, y, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1)
        return (ce * ((1 - pt) ** FOCAL_GAMMA)).mean()
    raise ValueError(f"unknown LOSS: {LOSS}")


def train_one_epoch(model, loader, optimizer, dev, ce_weight, aux_weight):
    model.train()
    losses, ys, preds = [], [], []
    aux_ce = nn.CrossEntropyLoss(weight=aux_weight) if AUX_5CLASS else None
    for feat, tab, y, y5, _, _ in loader:
        feat, tab, y, y5 = feat.to(dev), tab.to(dev), y.to(dev), y5.to(dev)
        optimizer.zero_grad()
        logits, aux_logits, _ = model(feat, tab)
        loss = binary_loss(logits, y, ce_weight)
        if aux_logits is not None:
            loss = loss + AUX_WEIGHT * aux_ce(aux_logits, y5)
        loss.backward()
        optimizer.step()
        prob = torch.softmax(logits, dim=1)[:, 1]
        losses.append(loss.item())
        ys.extend(y.cpu().tolist())
        preds.extend((prob.detach().cpu().numpy() >= 0.5).astype(int).tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def collect_predictions(model, loader, dev):
    model.eval()
    rows = []
    for feat, tab, y, _, case_ids, labels in loader:
        feat, tab = feat.to(dev), tab.to(dev)
        logits, _, weights = model(feat, tab)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        weights = weights.cpu().numpy()
        for i, case_id in enumerate(case_ids):
            rows.append(
                {
                    "case_id": case_id,
                    "true_label_5": labels[i],
                    "true_id": int(y[i].item()),
                    "prob_benign_neurogenic": float(prob[i, 0]),
                    "prob_nonbenign_actionable": float(prob[i, 1]),
                    "top_slice_index_in_bag": int(weights[i].argmax()) if weights.shape[1] else "",
                }
            )
    return rows


def add_predictions(rows, threshold):
    out = []
    for r in rows:
        rr = dict(r)
        pred = int(float(rr["prob_nonbenign_actionable"]) >= threshold)
        rr["threshold"] = threshold
        rr["pred_id"] = pred
        rr["pred_label"] = CLASS_NAMES[pred]
        out.append(rr)
    return out


def metrics_from_rows(rows, threshold):
    pred_rows = add_predictions(rows, threshold)
    ys = [int(r["true_id"]) for r in pred_rows]
    preds = [int(r["pred_id"]) for r in pred_rows]
    probs = [float(r["prob_nonbenign_actionable"]) for r in pred_rows]
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    out = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ppv": ppv,
        "npv": npv,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(set(ys)) == 2:
        out["auroc"] = roc_auc_score(ys, probs)
        out["average_precision"] = average_precision_score(ys, probs)
    return out, pred_rows


def choose_threshold(rows, mode, min_sensitivity):
    if mode == "fixed_05":
        return 0.5
    if mode == "sens90":
        mode, min_sensitivity = "screening", 0.90
    if mode == "sens85":
        mode, min_sensitivity = "screening", 0.85
    y = np.asarray([int(r["true_id"]) for r in rows])
    p = np.asarray([float(r["prob_nonbenign_actionable"]) for r in rows])
    thresholds = np.unique(np.r_[0.0, p, 1.0])
    best_t, best_score = 0.5, -1e18
    for t in thresholds:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        bacc = 0.5 * (sens + spec)
        if mode == "youden":
            score = sens + spec - 1.0
        elif mode == "balanced_accuracy":
            score = bacc
        elif mode == "screening":
            score = spec if sens >= min_sensitivity else sens - min_sensitivity - 1.0
        else:
            raise ValueError(f"unknown THRESHOLD_MODE: {mode}")
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def selection_score(metrics):
    if SELECT_METRIC == "macro_f1":
        return metrics["macro_f1"]
    if SELECT_METRIC == "balanced_accuracy":
        return metrics["balanced_accuracy"]
    if SELECT_METRIC == "youden":
        return metrics["sensitivity"] + metrics["specificity"] - 1.0
    if SELECT_METRIC == "screening":
        if metrics["sensitivity"] >= MIN_SENSITIVITY:
            return metrics["specificity"]
        return metrics["sensitivity"] - MIN_SENSITIVITY - 1.0
    raise ValueError(f"unknown SELECT_METRIC: {SELECT_METRIC}")


def subtype_metrics(rows, threshold):
    pred_rows = add_predictions(rows, threshold)
    out = []
    for label in LABEL5_NAMES:
        sub = [r for r in pred_rows if r["true_label_5"] == label]
        if not sub:
            continue
        target = binary_target({"label_5": label})
        correct = sum(int(r["pred_id"]) == target for r in sub)
        out.append(
            {
                "label_5": label,
                "n": len(sub),
                "binary_target": target,
                "binary_recall": correct / len(sub),
                "mean_prob_nonbenign": float(np.mean([float(r["prob_nonbenign_actionable"]) for r in sub])),
            }
        )
    return out


def loader_for(rows, tabular_encoder, train=False):
    sampler = make_sampler(rows) if train else None
    return DataLoader(
        BinaryBags(rows, tabular_encoder),
        batch_size=BATCH_SIZE,
        shuffle=(train and sampler is None),
        sampler=sampler,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()

    train_rows, val_rows, test_rows = read_rows("train"), read_rows("val"), read_rows("test")
    tabular_encoder = TabularEncoder(train_rows)
    train_loader = loader_for(train_rows, tabular_encoder, train=True)
    val_loader = loader_for(val_rows, tabular_encoder)
    test_loader = loader_for(test_rows, tabular_encoder)

    model = BinaryMILHead(tabular_encoder.dim).to(dev)
    ce_weight = class_weights(train_rows).to(dev)
    aux_weight = class5_weights(train_rows).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, best_threshold, log_rows = -1e18, 0.5, []
    print(
        f"binary_nonbenign fold={FOLD} train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)} "
        f"pooling={POOLING} fusion={FUSION} sampler={SAMPLER} loss={LOSS} threshold={THRESHOLD_MODE}",
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, dev, ce_weight, aux_weight)
        val_raw = collect_predictions(model, val_loader, dev)
        threshold = choose_threshold(val_raw, THRESHOLD_MODE, MIN_SENSITIVITY)
        val_metrics, _ = metrics_from_rows(val_raw, threshold)
        score = selection_score(val_metrics)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_accuracy": acc,
                "threshold": threshold,
                "selection_score": score,
                **val_metrics,
            }
        )
        if score > best_score:
            best_score, best_threshold = score, threshold
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_score": float(best_score),
                    "best_threshold": float(best_threshold),
                },
                OUT_DIR / "model_best.pt",
            )
        print(
            f"epoch {epoch}/{EPOCHS} loss={loss:.4f} train_acc={acc:.3f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.3f} val_sens={val_metrics['sensitivity']:.3f} "
            f"val_spec={val_metrics['specificity']:.3f} thr={threshold:.3f}",
            flush=True,
        )

    torch.save({"model_state": model.state_dict(), "epoch": EPOCHS, "best_score": float(best_score)}, OUT_DIR / "model_last.pt")
    checkpoint = torch.load(OUT_DIR / "model_best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_threshold = float(checkpoint["best_threshold"])

    val_raw = collect_predictions(model, val_loader, dev)
    test_raw = collect_predictions(model, test_loader, dev)
    val_metrics, val_pred = metrics_from_rows(val_raw, best_threshold)
    test_metrics, test_pred = metrics_from_rows(test_raw, best_threshold)
    val_metrics_05, val_pred_05 = metrics_from_rows(val_raw, 0.5)
    test_metrics_05, test_pred_05 = metrics_from_rows(test_raw, 0.5)

    write_rows(OUT_DIR / "train_log.csv", log_rows)
    write_rows(OUT_DIR / "val_predictions.csv", val_pred)
    write_rows(OUT_DIR / "test_predictions.csv", test_pred)
    write_rows(OUT_DIR / "val_predictions_fixed05.csv", val_pred_05)
    write_rows(OUT_DIR / "test_predictions_fixed05.csv", test_pred_05)
    write_rows(OUT_DIR / "test_subtype_metrics.csv", subtype_metrics(test_raw, best_threshold))
    (OUT_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "test_metrics_fixed05.json").write_text(json.dumps(test_metrics_05, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "thresholds.json").write_text(
        json.dumps(
            {
                "threshold_mode": THRESHOLD_MODE,
                "min_sensitivity": MIN_SENSITIVITY,
                "best_threshold": best_threshold,
                "best_epoch": int(checkpoint["epoch"]),
                "best_score": float(checkpoint["best_score"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "task": "benign_neurogenic_vs_nonbenign_actionable",
                "feature_name": FEATURE_NAME,
                "fold": FOLD,
                "pooling": POOLING,
                "fusion": FUSION,
                "sampler": SAMPLER,
                "loss": LOSS,
                "select_metric": SELECT_METRIC,
                "threshold_mode": THRESHOLD_MODE,
                "min_sensitivity": MIN_SENSITIVITY,
                "feature_norm": FEATURE_NORM,
                "aux_5class": AUX_5CLASS,
                "aux_weight": AUX_WEIGHT,
                "tabular": tabular_encoder.state(),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "seed": SEED,
                "class_names": CLASS_NAMES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"val": val_metrics, "test": test_metrics, "test_fixed05": test_metrics_05, "run": str(OUT_DIR)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
