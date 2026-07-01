#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_standard_v0"
CSV_PATH = DATASET_ROOT / "all.csv"
OUT_DIR = PROJECT_ROOT / "experiments" / "mil_resnet18_unfrozen_10ep"

NUM_CLASSES = 4
NUM_SLICES = 32
IMAGE_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE = False
SEED = 42

WINDOWS = [
    (-160.0, 240.0),
    (-200.0, 100.0),
    (-200.0, 400.0),
]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLASS_NAMES = ["肉瘤类", "良性神经源性肿瘤", "副神经节瘤", "淋巴瘤"]


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(split):
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["split"] == split]


def window_channel(x, low, high):
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def load_slice_bag(nifti_path):
    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    z = vol.shape[2]
    idx = np.linspace(0, z - 1, NUM_SLICES).round().astype(int)
    slices = vol[:, :, idx].transpose(2, 0, 1)
    channels = [window_channel(slices, low, high) for low, high in WINDOWS]
    x = np.stack(channels, axis=1).astype(np.float32)
    x = torch.from_numpy(x)
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class CTBags(Dataset):
    def __init__(self, split):
        self.rows = read_rows(split)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = load_slice_bag(DATASET_ROOT / r["image"])
        y = torch.tensor(int(r["label_4_id"]), dtype=torch.long)
        return x, y, r["group"], r["label_4"]


class AttentionMIL(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if FREEZE_BACKBONE:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        b, n, c, h, w = x.shape
        feat = self.backbone(x.view(b * n, c, h, w)).view(b, n, 512)
        weights = torch.softmax(self.attn(feat).squeeze(-1), dim=1)
        pooled = (feat * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled), weights


def class_weights(rows):
    counts = np.bincount([int(r["label_4_id"]) for r in rows], minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, dev):
    model.train()
    total_loss, ys, preds = 0.0, [], []
    for x, y, _, _ in loader:
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        ys.extend(y.cpu().tolist())
        preds.extend(logits.argmax(1).cpu().tolist())
    return total_loss / len(loader), accuracy_score(ys, preds)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    rows, ys, preds = [], [], []
    for x, y, groups, labels in loader:
        x = x.to(dev)
        logits, attn = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(prob.argmax())
        true = int(y.item())
        rows.append(
            {
                "group": groups[0],
                "true_label": labels[0],
                "true_id": true,
                "pred_label": CLASS_NAMES[pred],
                "pred_id": pred,
                "prob_肉瘤类": prob[0],
                "prob_良性神经源性肿瘤": prob[1],
                "prob_副神经节瘤": prob[2],
                "prob_淋巴瘤": prob[3],
                "top_slice_index_in_bag": int(attn.cpu().numpy()[0].argmax()),
            }
        )
        ys.append(true)
        preds.append(pred)
    metrics = {
        "accuracy": accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(ys, preds, labels=list(range(NUM_CLASSES))).tolist(),
        "classification_report": classification_report(
            ys, preds, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
    }
    return metrics, rows


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = device()
    print(f"device: {dev}")

    train_set = CTBags("train")
    test_set = CTBags("test")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = AttentionMIL().to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_set.rows).to(dev))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)

    log_rows = []
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        log_rows.append({"epoch": epoch, "train_loss": loss, "train_accuracy": acc})
        print(f"epoch {epoch}/{EPOCHS} train_loss={loss:.4f} train_acc={acc:.3f}")

    metrics, pred_rows = evaluate(model, test_loader, dev)
    torch.save(model.state_dict(), OUT_DIR / "model_last.pt")

    with (OUT_DIR / "train_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy"])
        w.writeheader()
        w.writerows(log_rows)

    with (OUT_DIR / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        w.writeheader()
        w.writerows(pred_rows)

    with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "confusion_matrix"]}, ensure_ascii=False, indent=2))
    print(f"outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
