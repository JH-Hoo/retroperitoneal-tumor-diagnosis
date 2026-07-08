#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "champion_flare23_25d_cache_15x224_minvox5000"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "pseudo_seg25d_clinical4_minvox5000"

CLINICAL4_CLASS_NAMES = ["sarcoma/GIST-like", "lymphoma", "PPGL", "benign neurogenic"]
DERIVED_BINARY_CLASS_NAMES = ["risk/workup", "benign-like"]
LABEL_5_TO_CLINICAL4 = {
    "肉瘤类": 0,
    "胃肠道间质瘤": 0,
    "淋巴瘤": 1,
    "PPGL": 2,
    "良性神经源性肿瘤": 3,
}
LABEL_5_ID_TO_CLINICAL4 = {0: 0, 4: 0, 3: 1, 2: 2, 1: 3}
CLASS_IDS = list(range(len(CLINICAL4_CLASS_NAMES)))
BENIGN_CLINICAL4_ID = 3
CT_MEAN = torch.tensor([0.485, 0.456], dtype=torch.float32).view(2, 1, 1)
CT_STD = torch.tensor([0.229, 0.224], dtype=torch.float32).view(2, 1, 1)


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_arg(text):
    text = str(text).lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def clinical4_id(row):
    if row.get("clinical4_id", "") != "":
        return int(row["clinical4_id"])
    label = row.get("label_5", "")
    if label in LABEL_5_TO_CLINICAL4:
        return LABEL_5_TO_CLINICAL4[label]
    label_5_id = row.get("label_5_id", "")
    if label_5_id != "":
        return LABEL_5_ID_TO_CLINICAL4[int(float(label_5_id))]
    raise ValueError(f"Cannot map row to clinical4 class: {row.get('group', '')}")


def binary_id_from_clinical4(cls):
    return 1 if int(cls) == BENIGN_CLINICAL4_ID else 0


def parse_semicolon_floats(text):
    if not text:
        return []
    return [float(x) for x in str(text).split(";") if str(x).strip()]


def stratified_val_split(rows, val_fraction, seed):
    rng = random.Random(seed)
    by_class = {}
    for row in rows:
        by_class.setdefault(clinical4_id(row), []).append(row)
    train, val = [], []
    for cls_rows in by_class.values():
        cls_rows = list(cls_rows)
        rng.shuffle(cls_rows)
        n_val = max(1, int(round(len(cls_rows) * val_fraction))) if len(cls_rows) > 1 else 0
        val.extend(cls_rows[:n_val])
        train.extend(cls_rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class PseudoSegSliceDataset(Dataset):
    def __init__(self, rows, cache_root, include_z=True):
        self.rows = rows
        self.cache_root = Path(cache_root)
        self.include_z = bool(include_z)
        self.items = [(ri, zi) for ri, row in enumerate(rows) for zi in range(int(row.get("num_slices", 15) or 15))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row_i, z_i = self.items[index]
        row = self.rows[row_i]
        tensor = torch.load(self.cache_root / row["tensor"], map_location="cpu").float().div(255.0)
        z_i = min(z_i, tensor.shape[0] - 1)
        sl = tensor[z_i]
        ct = (sl[0:2] - CT_MEAN) / CT_STD
        organ = sl[4:5]
        x_parts = [ct, organ]
        if self.include_z:
            z_norms = parse_semicolon_floats(row.get("selected_z_norm", ""))
            z_norm = z_norms[z_i] if z_i < len(z_norms) else float(z_i / max(tensor.shape[0] - 1, 1))
            x_parts.append(torch.full_like(organ, float(z_norm)))
        x = torch.cat(x_parts, dim=0)
        target = torch.zeros(sl.shape[1:], dtype=torch.long)
        target[sl[2] > 0.5] = clinical4_id(row) + 1
        return x, target


def make_case_input(tensor, row, include_z=True):
    tensor = tensor.float().div(255.0)
    ct = (tensor[:, 0:2] - CT_MEAN.view(1, 2, 1, 1)) / CT_STD.view(1, 2, 1, 1)
    organ = tensor[:, 4:5]
    parts = [ct, organ]
    if include_z:
        z_norms = parse_semicolon_floats(row.get("selected_z_norm", ""))
        if len(z_norms) != tensor.shape[0]:
            z_norms = np.linspace(0.0, 1.0, tensor.shape[0]).tolist()
        z = torch.tensor(z_norms, dtype=torch.float32).view(tensor.shape[0], 1, 1, 1).expand(-1, 1, tensor.shape[2], tensor.shape[3])
        parts.append(z)
    return torch.cat(parts, dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels, num_classes=5, base=16, dropout=0.1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(DoubleConv(base * 4, base * 8), nn.Dropout2d(dropout))
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def foreground_dice_loss(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)[:, 1:]
    one_hot = F.one_hot(target.clamp(0, 4), num_classes=5).permute(0, 3, 1, 2).float()[:, 1:]
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    present = one_hot.sum(dims) > 0
    if present.any():
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()


def class_weights_from_rows(rows, cache_root, bg_weight=0.05):
    counts = np.zeros(5, dtype=np.float64)
    cache_root = Path(cache_root)
    for row in rows:
        tensor = torch.load(cache_root / row["tensor"], map_location="cpu")
        tumor = tensor[:, 2] > 127
        pos = float(tumor.sum().item())
        total = float(tumor.numel())
        counts[0] += max(total - pos, 0.0)
        counts[clinical4_id(row) + 1] += pos
    pos_counts = np.maximum(counts[1:], 1.0)
    pos_total = pos_counts.sum()
    weights = np.ones(5, dtype=np.float32)
    weights[0] = float(bg_weight)
    weights[1:] = np.clip(pos_total / (4.0 * pos_counts), 0.25, 8.0)
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def metrics_dict(y, probs):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    labels = CLASS_IDS
    top2 = np.argsort(probs, axis=1)[:, -2:] if len(probs) else np.empty((0, 2), dtype=int)
    recall = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(np.mean([int(yy in tt) for yy, tt in zip(y, top2)])) if len(top2) else 0.0,
        "per_class_recall": {CLINICAL4_CLASS_NAMES[i]: float(recall[i]) for i in labels},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def binary_metrics_dict(y, probs):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = probs.argmax(axis=1)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "risk_workup_recall": float(recall[0]),
        "benign_like_recall": float(recall[1]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def plot_confusion(cm, labels, path, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "#1f2937")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def evaluate_cases(model, rows, cache_root, device, include_z=True, batch_size=8):
    model.eval()
    pred_rows, y_true, prob_rows, y_bin, prob_bin_rows = [], [], [], [], []
    fallback_count = 0
    for row in rows:
        tensor = torch.load(Path(cache_root) / row["tensor"], map_location="cpu")
        x = make_case_input(tensor, row, include_z=include_z)
        probs_all = []
        for start in range(0, x.shape[0], batch_size):
            logits = model(x[start : start + batch_size].to(device))
            probs_all.append(torch.softmax(logits, dim=1).cpu())
        probs = torch.cat(probs_all, dim=0)
        hard = probs.argmax(dim=1)
        hard_counts = torch.stack([(hard == (cls + 1)).sum() for cls in CLASS_IDS]).float()
        prob_mass = probs[:, 1:5].sum(dim=(0, 2, 3))
        used_fallback = int(hard_counts.sum().item() == 0)
        scores = prob_mass if used_fallback else hard_counts
        fallback_count += used_fallback
        score_np = scores.numpy().astype(np.float64)
        if score_np.sum() <= 0:
            score_np = np.ones(4, dtype=np.float64)
        prob4 = score_np / score_np.sum()
        order = np.argsort(prob4)[::-1]
        true_cls = clinical4_id(row)
        pred_cls = int(order[0])
        risk_score = float(prob4[0] + prob4[1] + prob4[2])
        benign_score = float(prob4[3])
        y_true.append(true_cls)
        prob_rows.append(prob4.tolist())
        y_bin.append(binary_id_from_clinical4(true_cls))
        prob_bin_rows.append([risk_score, benign_score])
        pred_rows.append(
            {
                "group": row["group"],
                "label_5": row.get("label_5", ""),
                "true_clinical4_label": CLINICAL4_CLASS_NAMES[true_cls],
                "true_clinical4_id": true_cls,
                "pred_clinical4_label": CLINICAL4_CLASS_NAMES[pred_cls],
                "pred_clinical4_id": pred_cls,
                "top1_clinical4_label": CLINICAL4_CLASS_NAMES[pred_cls],
                "top1_clinical4_prob": float(prob4[pred_cls]),
                "top2_clinical4_label": CLINICAL4_CLASS_NAMES[int(order[1])],
                "top2_clinical4_prob": float(prob4[int(order[1])]),
                "true_binary_label": DERIVED_BINARY_CLASS_NAMES[binary_id_from_clinical4(true_cls)],
                "true_binary_id": binary_id_from_clinical4(true_cls),
                "pred_binary_label": DERIVED_BINARY_CLASS_NAMES[int(benign_score > risk_score)],
                "pred_binary_id": int(benign_score > risk_score),
                "prob_risk_workup": risk_score,
                "prob_benign_like": benign_score,
                "used_prob_mass_fallback": used_fallback,
                "pred_foreground_pixels": int(hard_counts.sum().item()),
                "score_sarcoma_gist_like": float(score_np[0]),
                "score_lymphoma": float(score_np[1]),
                "score_ppgl": float(score_np[2]),
                "score_benign_neurogenic": float(score_np[3]),
            }
        )
    return pred_rows, y_true, prob_rows, y_bin, prob_bin_rows, fallback_count


def train_one_epoch(model, loader, optimizer, device, ce_weight, amp_enabled, dice_weight):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits, target, weight=ce_weight)
            if dice_weight > 0:
                loss = loss + float(dice_weight) * foreground_dice_loss(logits, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * x.size(0)
    return total_loss / max(len(loader.dataset), 1)


def main():
    parser = argparse.ArgumentParser(description="Pseudo label14 class-aware 2.5D U-Net segmentation-as-classification baseline.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--no-z-channel", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = device_arg(args.device)
    rows = [r for r in read_rows(args.cache_root / "all.csv") if r.get("cache_status") == "ok"]
    labels = [clinical4_id(r) for r in rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    include_z = not args.no_z_channel
    input_channels = 4 if include_z else 3
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_test_predictions, fold_details = [], []

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(rows, labels), 1):
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        train_val_rows = [rows[i] for i in train_val_idx]
        test_rows = [rows[i] for i in test_idx]
        train_rows, val_rows = stratified_val_split(train_val_rows, args.val_fraction, fold_seed)
        ce_weight, pixel_counts = class_weights_from_rows(train_rows, args.cache_root, args.background_weight)
        ce_weight = ce_weight.to(device)
        train_ds = PseudoSegSliceDataset(train_rows, args.cache_root, include_z=include_z)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = SmallUNet(input_channels, num_classes=5, base=args.base_channels, dropout=args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score, best_state, history = -1.0, None, []
        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device, ce_weight, args.amp, args.dice_weight)
            val_pred, yv, pv, yvb, pvb, val_fallbacks = evaluate_cases(
                model, val_rows, args.cache_root, device, include_z=include_z, batch_size=args.eval_batch_size
            )
            val_metrics = metrics_dict(yv, pv)
            score = val_metrics["macro_f1"] + 0.25 * val_metrics["top2_accuracy"]
            if score > best_score:
                best_score = score
                best_state = deepcopy(model.state_dict())
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_macro_f1": val_metrics["macro_f1"],
                    "val_top2_accuracy": val_metrics["top2_accuracy"],
                    "val_fallback_cases": val_fallbacks,
                }
            )
            print(
                f"[fold {fold}] epoch={epoch:02d} loss={loss:.4f} "
                f"val_macro_f1={val_metrics['macro_f1']:.3f} val_top2={val_metrics['top2_accuracy']:.3f}",
                flush=True,
            )
        if best_state is not None:
            model.load_state_dict(best_state)
        test_pred, yt, pt, ytb, ptb, test_fallbacks = evaluate_cases(
            model, test_rows, args.cache_root, device, include_z=include_z, batch_size=args.eval_batch_size
        )
        for row in test_pred:
            row["fold"] = fold
        all_test_predictions.extend(test_pred)
        fold_details.append(
            {
                "fold": fold,
                "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
                "pixel_class_counts_train": pixel_counts,
                "class_weights": ce_weight.detach().cpu().tolist(),
                "best_selection_score": best_score,
                "test_fallback_cases": test_fallbacks,
                "test_metrics": {
                    "clinical4": metrics_dict(yt, pt),
                    "derived_binary": binary_metrics_dict(ytb, ptb),
                },
                "history": history,
            }
        )

    y_all = [int(r["true_clinical4_id"]) for r in all_test_predictions]
    # Reconstruct normalized class probabilities directly from score columns.
    p_all = []
    for r in all_test_predictions:
        scores = np.asarray(
            [
                float(r["score_sarcoma_gist_like"]),
                float(r["score_lymphoma"]),
                float(r["score_ppgl"]),
                float(r["score_benign_neurogenic"]),
            ],
            dtype=np.float64,
        )
        scores = scores / max(float(scores.sum()), 1e-8)
        p_all.append(scores.tolist())
    y_bin_all = [int(r["true_binary_id"]) for r in all_test_predictions]
    p_bin_all = [[float(r["prob_risk_workup"]), float(r["prob_benign_like"])] for r in all_test_predictions]
    oof_clinical4 = metrics_dict(y_all, p_all)
    oof_binary = binary_metrics_dict(y_bin_all, p_bin_all)
    summary = {
        "task": "pseudo_label14_class_aware_segmentation_as_classification",
        "architecture": "Small 2D U-Net over cached 2.5D slices; label14 pixels assigned to the case clinical4 class",
        "cache_root": str(args.cache_root),
        "num_rows": len(rows),
        "folds": args.folds,
        "epochs": args.epochs,
        "input_channels": ["ct_soft", "ct_fat", "organ_union"] + (["z_position"] if include_z else []),
        "class_counts": Counter(CLINICAL4_CLASS_NAMES[i] for i in labels),
        "oof_metrics": {"clinical4": oof_clinical4, "derived_binary": oof_binary},
        "folds_detail": fold_details,
    }
    write_rows(args.out_dir / "oof_predictions.csv", all_test_predictions)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion(
        oof_clinical4["confusion_matrix"],
        CLINICAL4_CLASS_NAMES,
        args.out_dir / "pseudo_seg25d_clinical4_oof_confusion_matrix.png",
        f"Pseudo-seg clinical4 OOF confusion matrix (acc={oof_clinical4['accuracy']:.3f})",
    )
    plot_confusion(
        oof_binary["confusion_matrix"],
        DERIVED_BINARY_CLASS_NAMES,
        args.out_dir / "pseudo_seg25d_derived_binary_oof_confusion_matrix.png",
        f"Pseudo-seg derived binary OOF confusion matrix (acc={oof_binary['accuracy']:.3f})",
    )
    print(json.dumps(summary["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
