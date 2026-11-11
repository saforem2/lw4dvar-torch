import sys
import types

import torch

import forecast_model


def test_explicit_device_is_respected():
    assert forecast_model.resolve_device("cpu") == torch.device("cpu")


def test_auto_device_uses_ezpz(monkeypatch):
    ezpz = types.ModuleType("ezpz")
    ezpz.get_torch_device = lambda *, as_torch_device: torch.device("xpu:1")
    monkeypatch.setitem(sys.modules, "ezpz", ezpz)

    assert forecast_model.resolve_device("auto") == torch.device("xpu:1")


def test_none_device_uses_ezpz(monkeypatch):
    ezpz = types.ModuleType("ezpz")
    ezpz.get_torch_device = lambda *, as_torch_device: torch.device("cpu")
    monkeypatch.setitem(sys.modules, "ezpz", ezpz)

    assert forecast_model.resolve_device() == torch.device("cpu")
