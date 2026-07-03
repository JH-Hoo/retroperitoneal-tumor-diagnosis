#!/usr/bin/env python3
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "labels"
FOLD = 0
RUN_NAME = f"5class_groupcv_fold{FOLD}_resnet18_mil"
OUT_DIR = PROJECT_ROOT / "runs" / RUN_NAME
CONFIG_PATH = PROJECT_ROOT / "configs" / "5class_groupcv_fold0.yaml"

NUM_CLASSES = 5
EPOCHS = 12
BATCH_SIZE = 1
HEAD_LR = 3e-4
LAYER4_LR = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 20260703
FREEZE_BACKBONE_EPOCHS = 5

CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤", "胃肠道间质瘤"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(split):
    with (DATASET_ROOT / "splits" / f"fold_{FOLD}" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cached_bag(row):
    x = torch.load(PROJECT_ROOT / row["tensor"], map_location="cpu").float().div(255.0)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class CTBags(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return load_cached_bag(r), torch.tensor(int(r["label_5_id"])), r["case_id"], r["label_5"]


class AttentionMIL(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        b, n, c, h, w = x.shape
        feat = self.backbone(x.view(b * n, c, h, w)).view(b, n, 512)
        weights = torch.softmax(self.attn(feat).squeeze(-1), dim=1)
        pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled), weights


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def set_trainable(model, epoch):
    for p in model.backbone.parameters():
        p.requires_grad = False
    if epoch > FREEZE_BACKBONE_EPOCHS:
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
    freeze_batchnorm(model)


def class_weights(rows):
    counts = np.bincount([int(r["label_5_id"]) for r in rows], minlength=NUM_CLASSES)
    return torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    freeze_batchnorm(model)
    losses, ys, preds = [], [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
    return float(np.mean(losses)), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    rows, ys, preds = [], [], []
    for x, y, groups, labels in loader:
        x = x.to(dev)
        logits, attn = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        for i, group in enumerate(groups):
            pred = int(prob[i].argmax())
            true = int(y[i].item())
            row = {
                "case_id": group,
                "true_label": labels[i],
                "true_id": true,
                "pred_label": CLASS_NAMES[pred],
                "pred_id": pred,
                "top_slice_index_in_bag": int(attn.cpu().numpy()[i].argmax()),
            }
            for j, name in enumerate(CLASS_NAMES):
                row[f"prob_{name}"] = float(prob[i, j])
            rows.append(row)
            ys.append(true)
            preds.append(pred)
    return metrics_dict(ys, preds), rows


def metrics_dict(ys, preds):
    return {
        "accuracy": accuracy_score(ys, preds),
        "balanced_accuracy": balanced_accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "per_class_recall": recall_score(ys, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(ys, preds, labels=list(range(NUM_CLASSES))).tolist(),
    }


def save_checkpoint(path, model, epoch, best_score):
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "class_names": CLASS_NAMES,
            "fold": FOLD,
            "seed": SEED,
            "hyperparams": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "head_lr": HEAD_LR,
                "layer4_lr": LAYER4_LR,
                "weight_decay": WEIGHT_DECAY,
                "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
                "dataset_root": str(DATASET_ROOT),
                "run_name": RUN_NAME,
            },
        },
        path,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, OUT_DIR / "config.yaml")
    dev = device()
    print(f"device: {dev}", flush=True)

    train_rows, val_rows, test_rows = read_rows("train"), read_rows("val"), read_rows("test")
    train_loader = DataLoader(CTBags(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CTBags(val_rows), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(CTBags(test_rows), batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionMIL().to(dev)
    set_trainable(model, 1)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(dev))
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.attn.parameters()) + list(model.classifier.parameters()), "lr": HEAD_LR},
            {"params": model.backbone.layer4.parameters(), "lr": LAYER4_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    best_score, log_rows = -1.0, []

    for epoch in range(1, EPOCHS + 1):
        set_trainable(model, epoch)
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_metrics, _ = evaluate(model, val_loader, dev)
        log_rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": acc, **val_metrics})
        print(f"epoch {epoch}/{EPOCHS} loss={loss:.4f} train_acc={acc:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}", flush=True)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            save_checkpoint(OUT_DIR / "model_best.pt", model, epoch, best_score)

    save_checkpoint(OUT_DIR / "model_last.pt", model, EPOCHS, best_score)
    checkpoint = torch.load(OUT_DIR / "model_best.pt", map_location=dev)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, val_pred = evaluate(model, val_loader, dev)
    test_metrics, test_pred = evaluate(model, test_loader, dev)
    write_rows(OUT_DIR / "train_log.csv", log_rows)
    write_rows(OUT_DIR / "val_predictions.csv", val_pred)
    write_rows(OUT_DIR / "test_predictions.csv", test_pred)
    (OUT_DIR / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, ensure_ascii=False, indent=2), flush=True)
    print(f"outputs: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
