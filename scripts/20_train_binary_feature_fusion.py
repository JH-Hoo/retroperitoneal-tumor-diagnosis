#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


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
DROPOUT = float(os.environ.get("DROPOUT", "0.15"))
SEED = 20260704

CLASS_NAMES = ["benign_neurogenic", "nonbenign_actionable"]


def read_rows(split):
    with (LABEL_DIR / "splits" / f"fold_{FOLD}" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
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
        return feat, tab, torch.tensor(binary_target(row)), row["case_id"], row["label_5"]


class BinaryHead(nn.Module):
    def __init__(self, tab_dim):
        super().__init__()
        self.pooling = POOLING
        if POOLING == "attention":
            self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            image_dim = 512
        elif POOLING == "meanmax":
            image_dim = 1024
        elif POOLING in ["mean", "max"]:
            image_dim = 512
        else:
            raise ValueError(f"unknown POOLING: {POOLING}")
        if FUSION:
            self.tabular_branch = nn.Sequential(nn.Linear(tab_dim, 16), nn.ReLU(inplace=True))
            classifier_dim = image_dim + 16
        else:
            self.tabular_branch = None
            classifier_dim = image_dim
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(classifier_dim, 2))

    def pool(self, feat):
        if self.pooling == "mean":
            pooled = feat.mean(dim=1)
            weights = torch.full(feat.shape[:2], 1.0 / feat.shape[1], device=feat.device)
        elif self.pooling == "max":
            pooled = feat.max(dim=1).values
            weights = torch.zeros(feat.shape[:2], device=feat.device)
        elif self.pooling == "meanmax":
            pooled = torch.cat([feat.mean(dim=1), feat.max(dim=1).values], dim=1)
            weights = torch.full(feat.shape[:2], 1.0 / feat.shape[1], device=feat.device)
        elif self.pooling == "attention":
            weights = torch.softmax(self.attn(feat).squeeze(-1), dim=1)
            pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(self, feat, tab):
        image_feat, weights = self.pool(feat)
        if FUSION:
            image_feat = torch.cat([image_feat, self.tabular_branch(tab)], dim=1)
        return self.classifier(image_feat), weights


def class_weights(rows):
    counts = np.bincount([binary_target(r) for r in rows], minlength=2)
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    losses, ys, preds = [], [], []
    for feat, tab, y, _, _ in loader:
        feat, tab, y = feat.to(dev), tab.to(dev), y.to(dev)
        optimizer.zero_grad()
        logits, _ = model(feat, tab)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).detach().cpu().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    rows, ys, preds, probs = [], [], [], []
    for feat, tab, y, case_ids, labels in loader:
        feat, tab = feat.to(dev), tab.to(dev)
        logits, weights = model(feat, tab)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        weights = weights.cpu().numpy()
        for i, case_id in enumerate(case_ids):
            pred = int(prob[i].argmax())
            true = int(y[i].item())
            row = {
                "case_id": case_id,
                "true_label_5": labels[i],
                "true_id": true,
                "pred_label": CLASS_NAMES[pred],
                "pred_id": pred,
                "prob_benign_neurogenic": float(prob[i, 0]),
                "prob_nonbenign_actionable": float(prob[i, 1]),
                "top_slice_index_in_bag": int(weights[i].argmax()) if POOLING == "attention" else "",
            }
            rows.append(row)
            ys.append(true)
            preds.append(pred)
            probs.append(prob[i, 1])
    return metrics_dict(ys, preds, probs), rows


def metrics_dict(ys, preds, probs):
    tn, fp, fn, tp = confusion_matrix(ys, preds, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(set(ys)) == 2:
        out["auroc"] = roc_auc_score(ys, probs)
        out["average_precision"] = average_precision_score(ys, probs)
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()

    train_rows, val_rows, test_rows = read_rows("train"), read_rows("val"), read_rows("test")
    tabular_encoder = TabularEncoder(train_rows)
    train_loader = DataLoader(BinaryBags(train_rows, tabular_encoder), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BinaryBags(val_rows, tabular_encoder), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(BinaryBags(test_rows, tabular_encoder), batch_size=BATCH_SIZE, shuffle=False)

    model = BinaryHead(tabular_encoder.dim).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, log_rows = -1.0, []
    print(f"binary_nonbenign fold={FOLD} train/val/test={len(train_rows)}/{len(val_rows)}/{len(test_rows)} fusion={FUSION}", flush=True)

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_metrics, _ = evaluate(model, val_loader, dev)
        log_rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": acc, **val_metrics})
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_score": best_score}, OUT_DIR / "model_best.pt")
        print(f"epoch {epoch}/{EPOCHS} loss={loss:.4f} train_acc={acc:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}", flush=True)

    torch.save({"model_state": model.state_dict(), "epoch": EPOCHS, "best_score": best_score}, OUT_DIR / "model_last.pt")
    checkpoint = torch.load(OUT_DIR / "model_best.pt", map_location=dev)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, val_pred = evaluate(model, val_loader, dev)
    test_metrics, test_pred = evaluate(model, test_loader, dev)

    write_rows(OUT_DIR / "train_log.csv", log_rows)
    write_rows(OUT_DIR / "val_predictions.csv", val_pred)
    write_rows(OUT_DIR / "test_predictions.csv", test_pred)
    (OUT_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "task": "benign_neurogenic_vs_nonbenign_actionable",
                "feature_name": FEATURE_NAME,
                "fold": FOLD,
                "pooling": POOLING,
                "fusion": FUSION,
                "tabular": tabular_encoder.state(),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "class_names": CLASS_NAMES,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"val": val_metrics, "test": test_metrics, "run": str(OUT_DIR)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
