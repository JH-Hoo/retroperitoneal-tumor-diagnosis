#!/usr/bin/env python3
import tempfile
from pathlib import Path

import torch

import train_pseudo_seg25d_clinical4_cv as seg
import train_resnet25d_clinical4_cv as mil


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tensors").mkdir()
        rows = []
        for i, label in enumerate(["肉瘤类", "良性神经源性肿瘤", "PPGL", "淋巴瘤"]):
            group = f"SMOKE{i:04d}"
            tensor = torch.zeros(3, 5, 64, 64, dtype=torch.uint8)
            tensor[:, 0:2] = torch.randint(0, 255, (3, 2, 64, 64), dtype=torch.uint8)
            tensor[:, 2, 20:36, 20:36] = 255
            tensor[:, 4, 8:56, 8:56] = 255
            torch.save(tensor, root / "tensors" / f"{group}.pt")
            rows.append(
                {
                    "group": group,
                    "label_5": label,
                    "tensor": f"tensors/{group}.pt",
                    "cache_status": "ok",
                    "selected_z_norm": "0.2;0.5;0.8",
                    "tumor_voxels": "768",
                }
            )
        ds = mil.SliceBagDataset(rows, root, aux_scaler=None, channel_indices=mil.CHANNEL_SETS["all"])
        x, z, aux, y, group, label = ds[0]
        assert x.shape == (3, 5, 64, 64)
        model = mil.ResNet25DMIL(weights_name="none", in_channels=5, aux_dim=0)
        logits4, logits2, attn = model(x.unsqueeze(0), z.unsqueeze(0), aux.unsqueeze(0))
        assert logits4.shape == (1, 4)
        assert logits2.shape == (1, 2)
        assert attn.shape[:2] == (1, 3)

        seg_ds = seg.PseudoSegSliceDataset(rows, root, include_z=True)
        sx, sy = seg_ds[0]
        assert sx.shape == (4, 64, 64)
        assert sy.shape == (64, 64)
        seg_model = seg.SmallUNet(4, num_classes=5, base=4)
        out = seg_model(sx.unsqueeze(0))
        assert out.shape == (1, 5, 64, 64)
    print("smoke ok")


if __name__ == "__main__":
    main()
