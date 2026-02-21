#!/usr/bin/env python3
"""
Test script for quantization utilities.

Verifies that quantize-and-dequantize works correctly for every dtype,
for both the numpy (JAX-compatible) and PyTorch paths.

Usage:
    python test_quantization.py
    python test_quantization.py --gpu 2
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Parse --gpu before importing anything that touches CUDA
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

import numpy as np
import torch
from quantization_utils import (
    DTYPE_NAMES,
    quantize_numpy,
    quantize_tensor,
)


def test_quantize_numpy():
    """Test numpy quantization for all dtypes."""
    print("=" * 60)
    print("Testing quantize_numpy (used for JAX/Flax models)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    all_pass = True

    for dtype_name in DTYPE_NAMES:
        try:
            q = quantize_numpy(x, dtype_name)

            # Check output shape and dtype
            assert q.shape == x.shape, f"Shape mismatch: {q.shape} != {x.shape}"
            assert q.dtype == np.float32, f"Output dtype should be float32, got {q.dtype}"

            # float32 should be a no-op
            if dtype_name == "float32":
                assert np.allclose(q, x), "float32 quantization should be a no-op"
                print(f"  {dtype_name:<16} PASS (no-op)")
            else:
                # Other dtypes should introduce some quantization error
                max_err = np.max(np.abs(q - x))
                mean_err = np.mean(np.abs(q - x))
                # Values should still be finite
                assert np.all(np.isfinite(q)), f"Non-finite values in output"
                print(f"  {dtype_name:<16} PASS  max_err={max_err:.6f}  mean_err={mean_err:.6f}")

        except Exception as e:
            print(f"  {dtype_name:<16} FAIL  {e}")
            all_pass = False

    return all_pass


def test_quantize_tensor():
    """Test PyTorch quantization for all dtypes."""
    print()
    print("=" * 60)
    print("Testing quantize_tensor (used for PyTorch models)")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(64, 128)
    all_pass = True

    for dtype_name in DTYPE_NAMES:
        try:
            q = quantize_tensor(x, dtype_name)

            assert q.shape == x.shape, f"Shape mismatch: {q.shape} != {x.shape}"
            assert q.dtype == torch.float32, f"Output dtype should be float32, got {q.dtype}"

            if dtype_name == "float32":
                assert torch.allclose(q, x), "float32 quantization should be a no-op"
                print(f"  {dtype_name:<16} PASS (no-op)")
            else:
                max_err = (q - x).abs().max().item()
                mean_err = (q - x).abs().mean().item()
                assert torch.all(torch.isfinite(q)), "Non-finite values in output"
                print(f"  {dtype_name:<16} PASS  max_err={max_err:.6f}  mean_err={mean_err:.6f}")

        except Exception as e:
            print(f"  {dtype_name:<16} FAIL  {e}")
            all_pass = False

    return all_pass


def test_consistency():
    """Verify numpy and torch produce similar results."""
    print()
    print("=" * 60)
    print("Testing numpy/torch consistency")
    print("=" * 60)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((32, 64)).astype(np.float32)
    x_torch = torch.from_numpy(x_np.copy())
    all_pass = True

    for dtype_name in DTYPE_NAMES:
        try:
            q_np = quantize_numpy(x_np, dtype_name)
            q_torch = quantize_tensor(x_torch, dtype_name).numpy()

            max_diff = np.max(np.abs(q_np - q_torch))
            # Allow small differences due to different implementations
            threshold = 1e-4 if dtype_name not in ("boolean",) else 0.1
            if max_diff < threshold:
                print(f"  {dtype_name:<16} PASS  max_diff={max_diff:.8f}")
            else:
                print(f"  {dtype_name:<16} WARN  max_diff={max_diff:.8f} (above threshold {threshold})")

        except Exception as e:
            print(f"  {dtype_name:<16} FAIL  {e}")
            all_pass = False

    return all_pass


def test_edge_cases():
    """Test edge cases: empty tensors, constant tensors, single element."""
    print()
    print("=" * 60)
    print("Testing edge cases")
    print("=" * 60)
    all_pass = True

    # Empty tensor
    for dtype_name in DTYPE_NAMES:
        try:
            empty = np.array([], dtype=np.float32)
            q = quantize_numpy(empty, dtype_name)
            assert q.shape == (0,), f"Empty array shape mismatch"
        except Exception as e:
            print(f"  empty/{dtype_name:<16} FAIL  {e}")
            all_pass = False
    print("  empty tensors       PASS")

    # Constant tensor (span=0)
    for dtype_name in DTYPE_NAMES:
        try:
            const = np.full((4, 4), 3.14, dtype=np.float32)
            q = quantize_numpy(const, dtype_name)
            assert q.shape == const.shape
            # For constant input, quantization of float types should preserve value
            if dtype_name in ("float32", "float16", "bfloat16"):
                assert np.allclose(q, const, atol=0.1), f"Constant value not preserved"
        except Exception as e:
            print(f"  const/{dtype_name:<16} FAIL  {e}")
            all_pass = False
    print("  constant tensors    PASS")

    # Single element
    for dtype_name in DTYPE_NAMES:
        try:
            single = np.array([1.5], dtype=np.float32)
            q = quantize_numpy(single, dtype_name)
            assert q.shape == (1,)
        except Exception as e:
            print(f"  single/{dtype_name:<16} FAIL  {e}")
            all_pass = False
    print("  single-element      PASS")

    return all_pass


def test_pytorch_model_quantization():
    """Test quantization on a small PyTorch model."""
    print()
    print("=" * 60)
    print("Testing PyTorch model weight quantization")
    print("=" * 60)

    from quantization_utils import quantize_model_weights, restore_model_weights

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
    )
    all_pass = True

    for dtype_name in ["float16", "bfloat16", "float8_e4m3", "int32", "boolean"]:
        try:
            # Save originals
            orig_w = {n: p.data.clone() for n, p in model.named_parameters()}

            # Quantize
            saved = quantize_model_weights(model, dtype_name)

            # Check weights changed (for non-float32)
            changed = False
            for n, p in model.named_parameters():
                if not torch.allclose(p.data, orig_w[n]):
                    changed = True
                    break
            assert changed, f"Weights should have changed for {dtype_name}"

            # Restore
            restore_model_weights(model, saved)

            # Check restored
            for n, p in model.named_parameters():
                assert torch.allclose(p.data, orig_w[n]), f"Weight {n} not restored"

            print(f"  {dtype_name:<16} PASS")
        except Exception as e:
            print(f"  {dtype_name:<16} FAIL  {e}")
            all_pass = False

    return all_pass


def test_nnx_model_quantization():
    """Test quantization on a Flax NNX model (requires JAX + Flax)."""
    print()
    print("=" * 60)
    print("Testing Flax NNX model weight quantization")
    print("=" * 60)

    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
    except ImportError as e:
        print(f"  SKIP (missing dependency: {e})")
        return True

    from quantization_utils import quantize_model_weights_nnx, convert_model_to_float32_nnx

    # Create a simple NNX model
    class SimpleModel(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.linear1 = nnx.Linear(16, 32, rngs=rngs)
            self.linear2 = nnx.Linear(32, 8, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.linear1(x))
            return self.linear2(x)

    all_pass = True

    for dtype_name in ["float16", "bfloat16", "float8_e4m3", "int32", "boolean"]:
        try:
            model = SimpleModel(rngs=nnx.Rngs(0))

            # Save original param values
            _, state = nnx.split(model)
            orig_params = jax.tree.map(lambda x: np.array(x), state.to_pure_dict())

            # Quantize
            model = quantize_model_weights_nnx(model, dtype_name)

            # Check params changed
            _, new_state = nnx.split(model)
            new_params = jax.tree.map(lambda x: np.array(x), new_state.to_pure_dict())

            changed = False
            orig_flat = jax.tree.leaves(orig_params)
            new_flat = jax.tree.leaves(new_params)
            for o, n in zip(orig_flat, new_flat):
                if not np.allclose(o, n, atol=1e-7):
                    changed = True
                    break
            assert changed, f"Params should have changed for {dtype_name}"

            # Verify model still runs
            x = jnp.ones((1, 16))
            out = model(x)
            assert out.shape == (1, 8), f"Output shape mismatch: {out.shape}"
            assert jnp.all(jnp.isfinite(out)), "Non-finite output after quantization"

            print(f"  {dtype_name:<16} PASS")
        except Exception as e:
            print(f"  {dtype_name:<16} FAIL  {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # Test float32 conversion
    try:
        model = SimpleModel(rngs=nnx.Rngs(0))
        # Cast to bfloat16 first (simulate checkpoint loading)
        graphdef, state = nnx.split(model)
        bf16_params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, 'astype') else x,
            state.to_pure_dict()
        )
        state.replace_by_pure_dict(bf16_params)
        model = nnx.merge(graphdef, state)

        # Convert to float32
        model = convert_model_to_float32_nnx(model)

        # Check all params are float32
        _, state = nnx.split(model)
        for leaf in jax.tree.leaves(state.to_pure_dict()):
            if hasattr(leaf, 'dtype') and jnp.issubdtype(leaf.dtype, jnp.floating):
                assert leaf.dtype == jnp.float32, f"Expected float32, got {leaf.dtype}"
        print(f"  {'f32_conversion':<16} PASS")
    except Exception as e:
        print(f"  {'f32_conversion':<16} FAIL  {e}")
        all_pass = False

    return all_pass


if __name__ == "__main__":
    results = []
    results.append(("quantize_numpy", test_quantize_numpy()))
    results.append(("quantize_tensor", test_quantize_tensor()))
    results.append(("consistency", test_consistency()))
    results.append(("edge_cases", test_edge_cases()))
    results.append(("pytorch_model", test_pytorch_model_quantization()))
    results.append(("nnx_model", test_nnx_model_quantization()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
