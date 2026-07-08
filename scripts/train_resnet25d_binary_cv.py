#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "champion_flare23_25d_cache_15x224_minvox5000"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "champion_resnet25d_binary_minvox5000_cv5"

BINARY_CLASS_NAMES = ["良性", "恶性/风险组"]
PLOT_CLASS_NAMES = ["benign", "malignant/risk"]
BENIGN_LABEL_5_ID = 1
CT_MEAN = torch.tensor([0.485, 0.456], dtype=torch.float32).view(1, 2, 1, 1)
CT_STD = torch.tensor([0.229, 0.224], dtype=torch.float32).view(1, 2, 1, 1)
BASE_AUX_COLUMNS = [
    "no_tumor_label14",
    "z_peak_norm",
    "z_centroid_norm",
    "z_std_norm",
    "z_q10_norm",
    "z_q25_norm",
    "z_q50_norm",
    "z_q75_norm",
    "z_q90_norm",
    "tumor_z_slices",
    "tumor_z_extent_norm",
    "tumor_area_max_frac",
    "tumor_area_entropy",
    "tumor_voxels",
    "crop_x",
    "crop_y",
    "spacing_z_mm",
]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(text)


def binary_id(row):
    return 0 if int(row["label_5_id"]) == BENIGN_LABEL_5_ID else 1


def parse_float(value, default=0.0):
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def parse_semicolon_floats(text):
    if not text:
        return []
    return [float(x) for x in str(text).split(";") if str(x).strip()]


def parse_hist(row):
    vals = parse_semicolon_floats(row.get("z_hist", ""))
    return np.asarray(vals, dtype=np.float32)


def cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


class ZPriorScaler:
    def __init__(self, base_columns=None):
        self.base_columns = list(base_columns or BASE_AUX_COLUMNS)
        self.columns = []
        self.mean = None
        self.std = None
        self.prototypes = None

    def fit(self, rows):
        self.columns = [c for c in self.base_columns if c in rows[0]]
        hist_by_class = defaultdict(list)
        for row in rows:
            hist = parse_hist(row)
            if hist.size:
                hist_by_class[binary_id(row)].append(hist)
        self.prototypes = {}
        for cls in [0, 1]:
            hists = hist_by_class.get(cls, [])
            if hists:
                proto = np.stack(hists, axis=0).mean(axis=0)
                proto = proto / max(float(proto.sum()), 1e-8)
            else:
                proto = np.zeros_like(next(iter(hist_by_class.values()))[0]) if hist_by_class else np.zeros(16, dtype=np.float32)
            self.prototypes[cls] = proto.astype(np.float32)
        feats = np.stack([self.raw_features(row) for row in rows], axis=0)
        self.mean = feats.mean(axis=0)
        self.std = feats.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def raw_features(self, row):
        values = [parse_float(row.get(c, ""), 0.0) for c in self.columns]
        hist = parse_hist(row)
        if hist.size == 0:
            hist = np.zeros_like(self.prototypes[0])
        benign_proto = self.prototypes[0]
        risk_proto = self.prototypes[1]
        values.extend(
            [
                cosine(hist, benign_proto),
                cosine(hist, risk_proto),
                float(np.abs(hist - benign_proto).sum()),
                float(np.abs(hist - risk_proto).sum()),
            ]
        )
        return np.asarray(values, dtype=np.float32)

    def transform_row(self, row):
        feat = self.raw_features(row)
        return torch.from_numpy((feat - self.mean) / self.std).float()

    @property
    def dim(self):
        return len(self.columns) + 4

    def to_dict(self):
        return {
            "columns": self.columns + ["z_cosine_良性", "z_cosine_恶性风险组", "z_l1_良性", "z_l1_恶性风险组"],
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "class_z_prototypes": {BINARY_CLASS_NAMES[k]: v.tolist() for k, v in (self.prototypes or {}).items()},
        }


class SliceBagDataset(Dataset):
    def __init__(self, rows, cache_root, aux_scaler=None):
        self.rows = rows
        self.cache_root = Path(cache_root)
        self.aux_scaler = aux_scaler

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        x = torch.load(self.cache_root / row["tensor"], map_location="cpu").float().div(255.0)
        x[:, 0:2] = (x[:, 0:2] - CT_MEAN) / CT_STD
        z = parse_semicolon_floats(row.get("selected_z_norm", ""))
        if len(z) != x.shape[0]:
            z = np.linspace(0.0, 1.0, x.shape[0]).tolist()
        z = torch.tensor(z, dtype=torch.float32).view(x.shape[0], 1)
        aux = self.aux_scaler.transform_row(row) if self.aux_scaler else torch.empty(0, dtype=torch.float32)
        return x, z, aux, torch.tensor(binary_id(row), dtype=torch.long), row["group"], row["label_5"]


def make_loader(rows, cache_root, aux_scaler, batch_size, shuffle, num_workers):
    return DataLoader(
        SliceBagDataset(rows, cache_root, aux_scaler),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_resnet18_encoder(weights_name, in_channels, mask_channel_init):
    weights = None
    if weights_name == "imagenet":
        weights = ResNet18_Weights.DEFAULT
    elif weights_name != "none":
        raise ValueError("--weights must be 'imagenet' or 'none'")
    try:
        backbone = resnet18(weights=weights)
    except Exception as exc:
        if weights is None:
            raise
        raise RuntimeError(
            "Failed to load torchvision ImageNet ResNet18 weights. "
            "Check remote network access or pre-populate the torch hub checkpoint cache, "
            "then rerun with --weights imagenet. Use --weights none only for a no-pretraining ablation."
        ) from exc
    feature_dim = backbone.fc.in_features
    old_conv = backbone.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        if weights is not None:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.zero_()
            new_conv.weight[:, 0:1] = mean_weight
            if in_channels >= 2:
                new_conv.weight[:, 1:2] = mean_weight
            if in_channels > 2:
                if mask_channel_init == "mean":
                    new_conv.weight[:, 2:] = mean_weight
                elif mask_channel_init == "small":
                    new_conv.weight[:, 2:] = mean_weight * 0.1
                elif mask_channel_init == "zero":
                    new_conv.weight[:, 2:] = 0
                else:
                    raise ValueError("--mask-channel-init must be zero, small, or mean")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    backbone.conv1 = new_conv
    backbone.fc = nn.Identity()
    return backbone, feature_dim


class ResNet25DMIL(nn.Module):
    def __init__(
        self,
        weights_name="imagenet",
        in_channels=5,
        aux_dim=0,
        dropout=0.35,
        attn_dim=128,
        pos_dim=16,
        hidden_dim=256,
        mask_channel_init="zero",
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone, feature_dim = build_resnet18_encoder(weights_name, in_channels, mask_channel_init)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.pos_mlp = nn.Sequential(nn.Linear(1, pos_dim), nn.ReLU(inplace=True))
        bag_dim = feature_dim + pos_dim
        self.attn = nn.Sequential(nn.Linear(bag_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.classifier = nn.Sequential(
            nn.Linear(bag_dim + aux_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, z, aux=None):
        b, n, c, h, w = x.shape
        feat = self.backbone(x.view(b * n, c, h, w)).view(b, n, -1)
        pos = self.pos_mlp(z)
        feat = torch.cat([feat, pos], dim=-1)
        attn = torch.softmax(self.attn(feat).squeeze(-1), dim=1)
        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        if aux is not None and aux.numel():
            pooled = torch.cat([pooled, aux], dim=1)
        return self.classifier(pooled), attn


def class_weights(rows):
    counts = np.bincount([binary_id(r) for r in rows], minlength=2)
    safe = np.maximum(counts, 1)
    return torch.tensor(safe.sum() / (2 * safe), dtype=torch.float32)


def metrics_dict(ys, probs):
    pred = np.asarray(probs).argmax(axis=1)
    cm = confusion_matrix(ys, pred, labels=[0, 1])
    recall = recall_score(ys, pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(ys, pred),
        "balanced_accuracy": balanced_accuracy_score(ys, pred),
        "macro_f1": f1_score(ys, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, pred, average="weighted", zero_division=0),
        "benign_recall": float(recall[0]),
        "malignant_recall": float(recall[1]),
        "confusion_matrix": cm.tolist(),
    }


def stratified_val_split(rows, val_fraction, seed):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[binary_id(row)].append(row)
    train_rows, val_rows = [], []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        n_val = max(1, int(round(len(cls_rows) * val_fraction))) if len(cls_rows) >= 3 else 0
        val_rows.extend(cls_rows[:n_val])
        train_rows.extend(cls_rows[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def train_one_epoch(model, loader, criterion, optimizer, scaler, dev, use_amp):
    model.train()
    losses, ys, probs = [], [], []
    for x, z, aux, y, _, _ in loader:
        x, z, aux, y = x.to(dev), z.to(dev), aux.to(dev), y.to(dev)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, _ = model(x, z, aux)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
        ys.extend(y.detach().cpu().tolist())
        probs.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy().tolist())
    out = metrics_dict(ys, probs)
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


@torch.no_grad()
def evaluate(model, loader, dev, use_amp):
    model.eval()
    rows, ys, probs_all = [], [], []
    for x, z, aux, y, groups, labels in loader:
        x, z, aux = x.to(dev), z.to(dev), aux.to(dev)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits, attn = model(x, z, aux)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        attn_np = attn.cpu().numpy()
        for i, group in enumerate(groups):
            true = int(y[i].item())
            pred = int(prob[i].argmax())
            row = {
                "group": group,
                "label_5": labels[i],
                "true_binary_label": BINARY_CLASS_NAMES[true],
                "true_binary_id": true,
                "pred_binary_label": BINARY_CLASS_NAMES[pred],
                "pred_binary_id": pred,
                "prob_良性": float(prob[i, 0]),
                "prob_恶性风险组": float(prob[i, 1]),
                "top_slice_index_in_bag": int(attn_np[i].argmax()),
                "top_slice_attention": float(attn_np[i].max()),
            }
            rows.append(row)
            ys.append(true)
            probs_all.append(prob[i].tolist())
    return metrics_dict(ys, probs_all), rows


def add_fold(rows, fold):
    out = []
    for row in rows:
        rr = dict(row)
        rr["fold"] = fold
        out.append(rr)
    return out


def probs_from_prediction_rows(rows):
    y, probs = [], []
    for row in rows:
        y.append(int(row["true_binary_id"]))
        probs.append([float(row["prob_良性"]), float(row["prob_恶性风险组"])])
    return y, probs


def plot_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=PLOT_CLASS_NAMES)
    ax.set_yticks([0, 1], labels=PLOT_CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    threshold = np.asarray(cm).max() / 2 if np.asarray(cm).size else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i][j] > threshold else "#1f2937"
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color=color, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="2.5D ImageNet-pretrained ResNet MIL binary CV on FLARE23 label14-guided slices.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", type=device_arg, default=device_arg("auto"))
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--mask-channel-init", choices=["zero", "small", "mean"], default="zero")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-aux", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(args.cache_root / "all.csv") if r.get("label_5_id", "") != ""]
    rows = [r for r in rows if r.get("cache_status", "ok") == "ok"]
    y_all = np.asarray([binary_id(r) for r in rows], dtype=np.int64)
    dev = args.device
    use_amp = bool(args.amp and dev.type == "cuda")
    print(f"device: {dev} amp={use_amp}", flush=True)
    print(
        f"rows={len(rows)} folds={args.folds} weights={args.weights} "
        f"class_counts={dict(Counter(BINARY_CLASS_NAMES[y] for y in y_all))}",
        flush=True,
    )

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_test_predictions, fold_summaries = [], []
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(rows)), y_all), 1):
        fold_dir = args.out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        train_val_rows = [rows[i] for i in train_val_idx]
        test_rows = [rows[i] for i in test_idx]
        train_rows, val_rows = stratified_val_split(train_val_rows, args.val_fraction, fold_seed)
        aux_scaler = None if args.no_aux else ZPriorScaler().fit(train_rows)
        aux_dim = 0 if aux_scaler is None else aux_scaler.dim
        train_loader = make_loader(train_rows, args.cache_root, aux_scaler, args.batch_size, True, args.num_workers)
        val_loader = make_loader(val_rows, args.cache_root, aux_scaler, args.batch_size, False, args.num_workers)
        test_loader = make_loader(test_rows, args.cache_root, aux_scaler, args.batch_size, False, args.num_workers)

        model = ResNet25DMIL(
            weights_name=args.weights,
            in_channels=5,
            aux_dim=aux_dim,
            dropout=args.dropout,
            mask_channel_init=args.mask_channel_init,
            freeze_backbone=args.freeze_backbone,
        ).to(dev)
        criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(dev))
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_score, log_rows = -1.0, []
        print(
            f"fold {fold}/{args.folds} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
            f"train_counts={dict(Counter(BINARY_CLASS_NAMES[binary_id(r)] for r in train_rows))}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, dev, use_amp)
            val_metrics, _ = evaluate(model, val_loader, dev, use_amp)
            score = val_metrics.get("macro_f1", 0.0)
            row = {
                "fold": fold,
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            log_rows.append(row)
            print(
                f"fold {fold}/{args.folds} epoch {epoch}/{args.epochs} "
                f"loss={train_metrics.get('loss', 0):.4f} "
                f"train_macro_f1={train_metrics.get('macro_f1', 0):.3f} "
                f"val_macro_f1={val_metrics.get('macro_f1', 0):.3f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), fold_dir / "model_best.pt")

        model.load_state_dict(torch.load(fold_dir / "model_best.pt", map_location=dev))
        train_metrics, train_pred = evaluate(model, train_loader, dev, use_amp)
        val_metrics, val_pred = evaluate(model, val_loader, dev, use_amp)
        test_metrics, test_pred = evaluate(model, test_loader, dev, use_amp)
        test_pred = add_fold(test_pred, fold)
        all_test_predictions.extend(test_pred)
        torch.save(model.state_dict(), fold_dir / "model_last.pt")
        write_rows(fold_dir / "train_log.csv", log_rows)
        write_rows(fold_dir / "train_predictions.csv", add_fold(train_pred, fold))
        write_rows(fold_dir / "val_predictions.csv", add_fold(val_pred, fold))
        write_rows(fold_dir / "test_predictions.csv", test_pred)
        fold_summary = {
            "fold": fold,
            "splits": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
            "aux_scaler": aux_scaler.to_dict() if aux_scaler else None,
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
        }
        (fold_dir / "summary.json").write_text(json.dumps(fold_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_summaries.append(fold_summary)
        print(f"fold {fold} test {json.dumps(test_metrics, ensure_ascii=False)}", flush=True)

    y_oof, prob_oof = probs_from_prediction_rows(all_test_predictions)
    oof_metrics = metrics_dict(y_oof, prob_oof)
    write_rows(args.out_dir / "oof_predictions.csv", all_test_predictions)
    summary = {
        "task": "binary_benign_vs_malignant_risk_25d",
        "benign_definition": "label_5 == 良性神经源性肿瘤",
        "malignant_risk_definition": "肉瘤类 + PPGL + 淋巴瘤 + 胃肠道间质瘤",
        "cache_root": str(args.cache_root),
        "num_rows": len(rows),
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weights": args.weights,
        "mask_channel_init": args.mask_channel_init,
        "freeze_backbone": args.freeze_backbone,
        "use_aux": not args.no_aux,
        "class_counts": dict(Counter(BINARY_CLASS_NAMES[y] for y in y_all)),
        "oof_metrics": oof_metrics,
        "folds_detail": fold_summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_confusion_matrix(
        oof_metrics["confusion_matrix"],
        args.out_dir / "resnet25d_binary_oof_confusion_matrix.png",
        f"ResNet25D binary 5-fold OOF confusion matrix (acc={oof_metrics['accuracy']:.3f})",
    )
    print(json.dumps({"oof": oof_metrics}, ensure_ascii=False, indent=2), flush=True)
    print(f"outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
