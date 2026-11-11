"""Exercise ezpz device detection and autograd on a Polaris CUDA node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from forecast_model import resolve_device, synchronize_device


device = resolve_device()
if device.type != "cuda":
    raise RuntimeError(f"Expected device auto-selection to choose CUDA, got {device}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch reports no CUDA device on this Polaris node")

x = torch.arange(16, dtype=torch.float32, device=device, requires_grad=True)
loss = (x.sin() ** 2).mean()
loss.backward()
if x.grad is None or not torch.isfinite(x.grad).all():
    raise RuntimeError("CUDA forward/backward produced missing or non-finite gradients")
synchronize_device(device)
print(
    f"device={device} torch={torch.__version__} "
    f"cuda={torch.version.cuda} loss={loss.item():.6f} "
    f"grad_norm={x.grad.norm().item():.6f}"
)
