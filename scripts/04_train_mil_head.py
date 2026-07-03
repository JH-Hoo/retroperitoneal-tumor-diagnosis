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
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_NAME = os.environ.get("FEATURE_NAME", "features_cache_96slice_resnet18")
TASK = os.environ.get("TASK", "5class")
POOLING = os.environ.get("POOLING", "meanmax")
FOLD = int(os.environ.get("FOLD", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "80"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LR = float(os.environ.get("LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.0001"))
SEED = 20260703

LABEL_DIR = PROJECT_ROOT / "data" / "labels"
FEATURE_DIR = PROJECT_ROOT / "data" / FEATURE_NAME / "features"
RUN_NAME = os.environ.get("RUN_NAME", f"{TASK}_{FEATURE_NAME}_fold{FOLD}_{POOLING}")
OUT_DIR = PROJECT_ROOT / "runs" / RUN_NAME

TASKS = {
    "5class": ("label_5_id", None, ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]),
    "sarcoma": ("label_5", "肉瘤类", ["non_sarcoma", "sarcoma"]),
    "ppgl": ("label_5", "PPGL", ["non_ppgl", "ppgl"]),
    "lymphoma": ("label_5", "淋巴瘤", ["non_lymphoma", "lymphoma"]),
}


def read_rows(split):
    with (LABEL_DIR / "splits" / f"fold_{FOLD}" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def target(row):
    field, positive, _ = TASKS[TASK]
    if positive is None:
        return int(row[field])
    return int(row[field] == positive)


class FeatureBags(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        feat = torch.load(FEATURE_DIR / f"{row['case_id']}.pt", map_location="cpu").float()
        return feat, torch.tensor(target(row)), row["case_id"], row["label_5"]


class MILHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pooling = POOLING
        if POOLING == "attention":
            self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
            self.classifier = nn.Linear(512, num_classes)
        elif POOLING == "meanmax":
            self.classifier = nn.Linear(1024, num_classes)
        else:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, feat):
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
        else:
            raise ValueError(f"unknown pooling: {self.pooling}")
        return self.classifier(pooled), weights


def class_weights(rows, num_classes):
    counts = np.bincount([target(r) for r in rows], minlength=num_classes)
    return torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    losses, ys, preds = [], [], []
    for feat, y, _, _ in loader:
        feat, y = feat.to(dev), y.to(dev)
        optimizer.zero_grad()
        logits, _ = model(feat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, dev, class_names):
    model.eval()
    rows, ys, preds, probs = [], [], [], []
    for feat, y, case_ids, labels in loader:
        feat = feat.to(dev)
        logits, weights = model(feat)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        weights = weights.cpu().numpy()
        for i, case_id in enumerate(case_ids):
            pred = int(prob[i].argmax())
            true = int(y[i].item())
            row = {
                "case_id": case_id,
                "true_label": labels[i],
                "true_id": true,
                "pred_label": class_names[pred],
                "pred_id": pred,
                "top_slice_index_in_bag": int(weights[i].argmax()) if POOLING == "attention" else "",
            }
            for j, name in enumerate(class_names):
                row[f"prob_{name}"] = float(prob[i, j])
            rows.append(row)
            ys.append(true)
            preds.append(pred)
            probs.append(prob[i])
    return metrics_dict(ys, preds, np.asarray(probs), len(class_names)), rows


def metrics_dict(ys, preds, probs, num_classes):
    out = {
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "per_class_recall": recall_score(ys, preds, labels=list(range(num_classes)), average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(ys, preds, labels=list(range(num_classes))).tolist(),
    }
    if num_classes == 2 and len(set(ys)) == 2:
        out["auroc"] = roc_auc_score(ys, probs[:, 1])
        out["average_precision"] = average_precision_score(ys, probs[:, 1])
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    _, _, class_names = TASKS[TASK]
    num_classes = len(class_names)

    train_rows, val_rows, test_rows = read_rows("train"), read_rows("val"), read_rows("test")
    train_loader = DataLoader(FeatureBags(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FeatureBags(val_rows), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(FeatureBags(test_rows), batch_size=BATCH_SIZE, shuffle=False)

    model = MILHead(num_classes).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, num_classes).to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, log_rows = -1.0, []

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_metrics, _ = evaluate(model, val_loader, dev, class_names)
        log_rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": acc, **val_metrics})
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "best_score": best_score}, OUT_DIR / "model_best.pt")
        print(f"epoch {epoch}/{EPOCHS} loss={loss:.4f} train_acc={acc:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}", flush=True)

    torch.save({"model_state": model.state_dict(), "epoch": EPOCHS, "best_score": best_score}, OUT_DIR / "model_last.pt")
    checkpoint = torch.load(OUT_DIR / "model_best.pt", map_location=dev)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, val_pred = evaluate(model, val_loader, dev, class_names)
    test_metrics, test_pred = evaluate(model, test_loader, dev, class_names)

    write_rows(OUT_DIR / "train_log.csv", log_rows)
    write_rows(OUT_DIR / "val_predictions.csv", val_pred)
    write_rows(OUT_DIR / "test_predictions.csv", test_pred)
    (OUT_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "pooling": POOLING,
                "fold": FOLD,
                "feature_name": FEATURE_NAME,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "class_names": class_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"val": val_metrics, "test": test_metrics, "run": str(OUT_DIR)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
