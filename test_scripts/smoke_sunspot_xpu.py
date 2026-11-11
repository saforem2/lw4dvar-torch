"""Exercise ezpz device detection and autograd on a Sunspot XPU compute node."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from forecast_model import resolve_device


device = resolve_device()
if device.type != "xpu":
    raise RuntimeError(f"Expected ezpz to select a Sunspot XPU, got {device}")

x = torch.arange(16, dtype=torch.float32, device=device, requires_grad=True)
loss = (x.sin() ** 2).mean()
loss.backward()
if x.grad is None or not torch.isfinite(x.grad).all():
    raise RuntimeError("XPU forward/backward produced missing or non-finite gradients")
from forecast_model import synchronize_device
synchronize_device(device)
print(f"device={device} torch={torch.__version__} loss={loss.item():.6f} grad_norm={x.grad.norm().item():.6f}")
