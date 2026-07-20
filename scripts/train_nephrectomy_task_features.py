#!/usr/bin/env python3
"""Train two fold-specific 3-D auxiliary tasks and extract 128 deep features.

Yang et al. used stage and grade supervision.  Reliable stage/grade labels are
not available in this cohort, so the same task-oriented representation idea is
adapted to two observed, non-nephrectomy targets:

* pathology phenotype: sarcoma-like vs other labeled tumor types;
* anatomy phenotype: tumor surface within 10 mm of the nearest kidney.

Each task has its own compact 3-D residual encoder with a 64-dimensional hidden
layer.  For every outer nephrectomy fold, auxiliary training excludes the
outer-test patients.  Unknown nephrectomy cases may supplement auxiliary
training because their surgical outcome is never used.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    keys = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def known_rows(rows):
    return [
        row
        for row in rows
        if row.get("feature_status") == "ok" and row.get("nephrectomy", "") in {"0", "1", "0.0", "1.0"}
    ]


def make_outer_splits(rows, folds, seed):
    ordered = sorted(rows, key=lambda row: row["case_id"])
    y = np.asarray([int(float(row["nephrectomy"])) for row in ordered], dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        result.append(
            {
                "fold": fold,
                "train_case_ids": [ordered[i]["case_id"] for i in train_idx],
                "test_case_ids": [ordered[i]["case_id"] for i in test_idx],
            }
        )
    return result


class RoiDataset(Dataset):
    def __init__(self, rows, feature_root: Path, task, augment=False):
        self.rows = rows
        self.feature_root = feature_root
        self.task = task
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def label(self, row):
        if self.task == "pathology":
            return int(row.get("label_4") == "肉瘤类")
        if self.task == "proximity":
            return int(as_float(row.get("geo_min_surface_distance_mm")) <= 10.0)
        raise ValueError(self.task)

    def __getitem__(self, index):
        row = self.rows[index]
        payload = np.load(self.feature_root / row["roi_path"])
        ct = payload["ct"].astype(np.float32)
        tumor = payload["tumor"].astype(np.float32)
        kidney = payload["kidney"].astype(np.float32)
        interface = payload["interface"].astype(np.float32)
        x = np.stack([ct, tumor, kidney, interface], axis=0)
        if self.augment:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    x = np.flip(x, axis=axis).copy()
            if random.random() < 0.5:
                x[0] += np.random.normal(0.0, 0.03, size=x[0].shape).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.tensor(self.label(row), dtype=torch.long),
            row["case_id"],
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class TaskResNet3D(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 32, 2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.hidden = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        features = self.hidden(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


def task_rows(all_rows, task, excluded_ids):
    rows = [row for row in all_rows if row.get("feature_status") == "ok" and row["case_id"] not in excluded_ids]
    if task == "pathology":
        rows = [row for row in rows if row.get("label_4") not in {"", "unlabeled", "None"}]
    return rows


def stratified_train_val(rows, task, seed, val_fraction=0.15):
    dataset = RoiDataset(rows, Path("."), task)
    y = np.asarray([dataset.label(row) for row in rows], dtype=int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 2:
        return rows, rows
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(rows)), y))
    return [rows[i] for i in train_idx], [rows[i] for i in val_idx]


def loader(rows, feature_root, task, batch_size, shuffle, workers, augment=False):
    return DataLoader(
        RoiDataset(rows, feature_root, task, augment=augment),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def validation_loss(model, data_loader, criterion, device, amp):
    model.eval()
    losses = []
    for x, y, _ in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            losses.append(float(criterion(model(x), y).item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_task(rows, feature_root, task, out_path, args, fold_seed):
    train_rows, val_rows = stratified_train_val(rows, task, fold_seed)
    train_ds = RoiDataset(train_rows, feature_root, task)
    counts = np.bincount([train_ds.label(row) for row in train_rows], minlength=2)
    if counts.min() == 0:
        raise RuntimeError(f"task {task} has one class: {counts.tolist()}")
    weights = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=args.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = TaskResNet3D(hidden_dim=64, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    train_loader = loader(train_rows, feature_root, task, args.batch_size, True, args.workers, augment=True)
    val_loader = loader(val_rows, feature_root, task, args.batch_size, False, args.workers)

    best_loss = float("inf")
    history = []
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device.type, enabled=args.amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        val_loss = validation_loss(model, val_loader, criterion, args.device, args.amp)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        print(f"task={task} epoch={epoch} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            patience_left = args.patience
            torch.save(model.state_dict(), out_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(torch.load(out_path, map_location=args.device))
    return model, counts.tolist(), history


@torch.no_grad()
def extract_embeddings(model, rows, feature_root, task, args, prefix):
    data_loader = loader(rows, feature_root, task, args.batch_size, False, args.workers)
    model.eval()
    output = {}
    for x, _, case_ids in data_loader:
        x = x.to(args.device)
        with torch.autocast(device_type=args.device.type, enabled=args.amp):
            _, features = model(x, return_features=True)
        values = features.float().cpu().numpy()
        for case_id, vector in zip(case_ids, values):
            output[case_id] = {f"deep_{prefix}_{i:02d}": float(v) for i, v in enumerate(vector)}
    return output


def combine_embeddings(case_rows, embeddings_by_task):
    output = []
    for row in case_rows:
        combined = {"case_id": row["case_id"], "nephrectomy": int(float(row["nephrectomy"]))}
        for embeddings in embeddings_by_task:
            combined.update(embeddings[row["case_id"]])
        output.append(combined)
    return output


def device_arg(text):
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0, help="optional smoke-test limit")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=device_arg, default=device_arg("auto"))
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    args.amp = bool(args.amp and args.device.type == "cuda")
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_csv(args.feature_root / "features.csv")
    outcome_rows = known_rows(all_rows)
    splits = make_outer_splits(outcome_rows, args.folds, args.seed)
    if args.max_folds:
        splits = splits[: args.max_folds]
    (args.out_dir / "outer_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    by_id = {row["case_id"]: row for row in all_rows}
    summaries = []

    for split in splits:
        fold = split["fold"]
        fold_dir = args.out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_case_rows = [by_id[case_id] for case_id in split["train_case_ids"]]
        test_case_rows = [by_id[case_id] for case_id in split["test_case_ids"]]
        train_embeddings = []
        test_embeddings = []
        fold_summary = {"fold": fold, "tasks": {}}
        excluded_ids = set(split["test_case_ids"])

        for task, prefix in (("pathology", "pathology"), ("proximity", "proximity")):
            fold_seed = args.seed + fold * 100 + (0 if task == "pathology" else 1)
            set_seed(fold_seed)
            auxiliary_rows = task_rows(all_rows, task, excluded_ids)
            model_path = fold_dir / f"{task}_best.pt"
            model, counts, history = train_task(
                auxiliary_rows, args.feature_root, task, model_path, args, fold_seed
            )
            train_embeddings.append(
                extract_embeddings(model, train_case_rows, args.feature_root, task, args, prefix)
            )
            test_embeddings.append(
                extract_embeddings(model, test_case_rows, args.feature_root, task, args, prefix)
            )
            fold_summary["tasks"][task] = {
                "auxiliary_cases": len(auxiliary_rows),
                "class_counts": counts,
                "best_val_loss": min(item["val_loss"] for item in history),
                "epochs_completed": len(history),
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_csv(fold_dir / "train_deep_features.csv", combine_embeddings(train_case_rows, train_embeddings))
        write_csv(fold_dir / "test_deep_features.csv", combine_embeddings(test_case_rows, test_embeddings))
        summaries.append(fold_summary)
        print(json.dumps(fold_summary, ensure_ascii=False), flush=True)

    payload = {
        "method": "two task-oriented 3-D residual encoders, 64 features each",
        "tasks": {
            "pathology": "sarcoma-like vs other labeled pathology phenotype",
            "proximity": "nearest tumor-kidney surface distance <=10 mm",
        },
        "outer_test_excluded_from_auxiliary_training": True,
        "unknown_nephrectomy_cases_allowed_for_auxiliary_training": True,
        "folds": summaries,
    }
    (args.out_dir / "deep_feature_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
