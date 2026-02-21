"""
Quantization utilities for FP precision benchmarking.
Supports both PyTorch (nn.Module) and JAX/Flax NNX models.
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DTYPE_NAMES = [
    "float32", "float16", "bfloat16",
    "float8_e4m3", "float8_e5m2",
    "int32", "uint32", "boolean",
]

QUANTIZATION_MODES = ["weights_only", "activations_only", "both"]


# ═══════════════════════════════════════════════════════════════════════════════
# Core quantization (numpy-based, framework-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_numpy(x: np.ndarray, dtype_name: str) -> np.ndarray:
    """Quantize a numpy array to the target dtype and convert back to float32.

    For float dtypes this is a simple cast-and-back.
    For integer / boolean dtypes we use min-max linear quantization.
    """
    if x.size == 0:
        return x.copy()

    src = x.astype(np.float32)

    # ── Float types: direct cast round-trip ──────────────────────────────────
    if dtype_name == "float32":
        return src.copy()
    if dtype_name == "float16":
        return src.astype(np.float16).astype(np.float32)
    if dtype_name == "bfloat16":
        import ml_dtypes
        return src.astype(ml_dtypes.bfloat16).astype(np.float32)
    if dtype_name == "float8_e4m3":
        import ml_dtypes
        # np.finfo doesn't work with ml_dtypes float8 on older numpy
        clamped = np.clip(src, -448.0, 448.0)
        return clamped.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    if dtype_name == "float8_e5m2":
        import ml_dtypes
        clamped = np.clip(src, -57344.0, 57344.0)
        return clamped.astype(ml_dtypes.float8_e5m2fnuz).astype(np.float32)

    # ── Integer / boolean types: linear quantization ─────────────────────────
    xmin = src.min()
    xmax = src.max()
    span = xmax - xmin
    if span == 0:
        return src.copy()

    if dtype_name == "boolean":
        median = np.median(src)
        return np.where(src > median, xmax, xmin).astype(np.float32)

    if dtype_name == "int32":
        qmin, qmax = -(2**31), 2**31 - 1
        scale = span / (qmax - qmin)
        zero_point = round(float(xmin / scale)) - qmin
        quantized = np.clip(np.round(src / scale) + zero_point, qmin, qmax).astype(np.int32)
        return ((quantized.astype(np.float32) - zero_point) * scale)

    if dtype_name == "uint32":
        qmin, qmax = 0, 2**32 - 1
        scale = span / (qmax - qmin)
        quantized = np.clip(np.round((src - xmin) / scale), 0, qmax).astype(np.int64)
        return (quantized.astype(np.float32) * scale + xmin)

    raise ValueError(f"Unsupported dtype: {dtype_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch quantization
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_tensor(x: torch.Tensor, dtype_name: str) -> torch.Tensor:
    """Quantize a torch tensor to the target dtype and convert back to float32."""
    if x.numel() == 0:
        return x.clone()

    src = x.float()

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
    """Quantize all PyTorch model weights in-place. Returns originals for restoration."""
    originals = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            originals[name] = param.data.cpu().clone()
            param.data = quantize_tensor(param.data, dtype_name).to(param.device)
    torch.cuda.empty_cache()
    return originals


def restore_model_weights(model: nn.Module, originals: dict):
    """Restore original PyTorch weights after quantization."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data = originals[name].to(param.device)
    del originals
    torch.cuda.empty_cache()


class ActivationQuantizer:
    """Registers forward hooks on all leaf PyTorch modules to quantize their outputs."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# JAX / Flax NNX quantization
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_model_weights_nnx(model, dtype_name: str):
    """Quantize all Flax NNX model weights.

    Returns the new quantized model (nnx.split dehydrates the original,
    so we merge into a fresh model rather than trying to update in-place).

    Memory-efficient: quantized results are cast back to the original dtype
    (e.g. bfloat16) before uploading to GPU.
    """
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    graphdef, state = nnx.split(model)
    params_dict = state.to_pure_dict()

    # Flatten and drop the nested dict so old arrays can be GC'd as we replace them
    flat, treedef = jax.tree_util.tree_flatten(params_dict)
    del params_dict

    count = 0
    for i in range(len(flat)):
        x = flat[i]
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            orig_dtype = x.dtype
            np_val = np.asarray(x)
            q_val = quantize_numpy(np_val, dtype_name)
            # Cast back to original numpy dtype to avoid doubling GPU memory
            if orig_dtype == jnp.bfloat16:
                q_val = q_val.astype(ml_dtypes.bfloat16)
            elif orig_dtype == jnp.float16:
                q_val = q_val.astype(np.float16)
            flat[i] = jnp.array(q_val)
            count += 1

    quantized_params = treedef.unflatten(flat)
    del flat

    state.replace_by_pure_dict(quantized_params)
    new_model = nnx.merge(graphdef, state)

    logger.info(f"Quantized {count} parameter arrays to {dtype_name}")
    return new_model


def convert_model_to_float32_nnx(model):
    """Convert all Flax NNX model params to float32. Returns the new model."""
    from flax import nnx
    import jax
    import jax.numpy as jnp

    graphdef, state = nnx.split(model)
    params_dict = state.to_pure_dict()

    def to_f32(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.float32)
        return x

    f32_params = jax.tree.map(to_f32, params_dict)
    state.replace_by_pure_dict(f32_params)
    return nnx.merge(graphdef, state)
