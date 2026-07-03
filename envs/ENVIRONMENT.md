# Python/CUDA Environments

This project keeps GPU environments separate from the code. The training scripts only require PyTorch, torchvision, nibabel, scikit-learn, matplotlib, numpy, and pandas.

## Recommended Portable Environment For RTX 30 Series

Use `environment-cu118.yml` on RTX 3060/3070/3080/3090, A4000/A5000/A6000, and most CUDA 11.x servers.

```bash
conda env create -f envs/environment-cu118.yml
conda activate rtp-mil-cu118
python - <<'PY'
import torch, torchvision
print(torch.__version__, torchvision.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

Equivalent pip install:

```bash
python -m venv .venv-cu118
source .venv-cu118/bin/activate
pip install -r envs/requirements-cu118.txt
```

## Current RTX 5090 D Environment

The current SeeTaCloud 5090 D host uses:

```text
Python 3.12.3
torch 2.8.0+cu128
torchvision 0.23.0+cu128
NVIDIA driver 595.71.05
CUDA runtime reported by driver: 13.2
```

Do not replace this base environment with CUDA 11.8 on the 5090 D host. RTX 50-series cards need newer PyTorch/CUDA builds, while CUDA 11.8 is the portable target for older RTX 30-series machines.

## Version Choice

- `cu118`: best portability target for older NVIDIA servers and RTX 30-series cards.
- `cu121`: also fine for RTX 30/40-series if the server driver is new enough, but it is less conservative than `cu118`.
- `cu128`: useful for very new cards such as RTX 5090 D; not the best default for older shared servers.
