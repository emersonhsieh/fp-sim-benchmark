"""
Quantization utilities for FP precision benchmarking.
Extracted from fp-benchmarks/benchmark.py for reuse across projects.
"""

import torch
import torch.nn as nn


DTYPE_NAMES = [
    "float32", "float16", "bfloat16",
    "float8_e4m3", "float8_e5m2",
    "int32", "uint32", "boolean",
]

QUANTIZATION_MODES = ["weights_only", "activations_only", "both"]


def quantize_tensor(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a tensor to the target dtype and convert back to float32.

    For float dtypes this is a simple cast-and-back.
    For integer / boolean dtypes we use min-max linear quantization.
    """
    if x.numel() == 0:
        return x.clone()

    src = x.float()

    # ── Float types: direct cast round-trip ──────────────────────────────────
    if dtype_name == "float32":
        return src.clone()
    if dtype_name == "float16":
        return src.to(torch.float16).float()
    if dtype_name == "bfloat16":
        return src.to(torch.bfloat16).float()
    if dtype_name == "float8_e4m3":
        finfo = torch.finfo(torch.float8_e4m3fn)
        clamped = src.clamp(finfo.min, finfo.max)
        return clamped.to(torch.float8_e4m3fn).float()
    if dtype_name == "float8_e5m2":
        finfo = torch.finfo(torch.float8_e5m2fnuz)
        clamped = src.clamp(finfo.min, finfo.max)
        return clamped.to(torch.float8_e5m2fnuz).float()

    # ── Integer / boolean types: linear quantization ─────────────────────────
    xmin = src.min()
    xmax = src.max()
    span = xmax - xmin
    if span == 0:
        return src.clone()

    if dtype_name == "boolean":
        median = src.median()
        return (src > median).float() * xmax + (src <= median).float() * xmin

    if dtype_name == "int32":
        qmin, qmax = -(2**31), 2**31 - 1
        scale = span / (qmax - qmin)
        zero_point = round((xmin / scale).item()) - qmin
        quantized = torch.clamp(torch.round(src / scale) + zero_point, qmin, qmax).to(torch.int32)
        return ((quantized.float() - zero_point) * scale)

    if dtype_name == "uint32":
        qmin, qmax = 0, 2**32 - 1
        scale = span / (qmax - qmin)
        quantized = torch.clamp(torch.round((src - xmin) / scale), 0, qmax).to(torch.int64)
        return (quantized.float() * scale + xmin)

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def quantize_model_weights(model: nn.Module, dtype_name: str) -> dict:
    """Quantize all model weights in-place. Returns dict of original weights (on CPU) for restoration."""
    originals = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            originals[name] = param.data.cpu().clone()
            param.data = quantize_tensor(param.data, dtype_name).to(param.device)
    torch.cuda.empty_cache()
    return originals


def restore_model_weights(model: nn.Module, originals: dict):
    """Restore original weights after quantization."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = originals[name].to(param.device)
    del originals
    torch.cuda.empty_cache()


class ActivationQuantizer:
    """Registers forward hooks on all leaf modules to quantize their outputs."""

    def __init__(self, model: nn.Module, dtype_name: str):
        self.dtype_name = dtype_name
        self.hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def _make_hook(self, name):
        dtype_name = self.dtype_name

        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.is_floating_point():
                return quantize_tensor(output, dtype_name).to(output.device)
            elif isinstance(output, tuple):
                return tuple(
                    quantize_tensor(o, dtype_name).to(o.device)
                    if isinstance(o, torch.Tensor) and o.is_floating_point()
                    else o
                    for o in output
                )
            return output

        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
