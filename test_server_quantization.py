#!/usr/bin/env python3
"""
Integration test for serve_quantized_policy.py quantization.

Loads the actual Pi0 model, applies quantization for each dtype,
and verifies the weights actually changed. Optionally starts the
server and makes a test inference via WebSocket.

Usage (from the openpi directory):
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        uv run python /path/to/test_server_quantization.py \
        --checkpoint gs://openpi-assets/checkpoints/pi05_droid_jointpos \
        --config pi05_droid_jointpos_polaris

    # Or with OPENPI_DATA_HOME if checkpoint is already cached:
    CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    OPENPI_DATA_HOME=/scratch/emersonhsieh/data/openpi \
        uv run python /path/to/test_server_quantization.py \
        --checkpoint gs://openpi-assets/checkpoints/pi05_droid_jointpos \
        --config pi05_droid_jointpos_polaris
"""

import sys
import os
import argparse
import logging
import time

# Add script directory for quantization_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# Monkey-patch torch.compile before importing openpi models
import openpi.models_pytorch.pi0_pytorch as pi0_mod
_original_init = pi0_mod.PI0Pytorch.__init__
def _patched_init(self, config):
    _orig_compile = torch.compile
    torch.compile = lambda fn, **kw: fn
    try:
        _original_init(self, config)
    finally:
        torch.compile = _orig_compile
pi0_mod.PI0Pytorch.__init__ = _patched_init

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

from quantization_utils import (
    quantize_model_weights, quantize_model_weights_nnx,
    convert_model_to_float32_nnx, DTYPE_NAMES,
)


def get_param_snapshot(policy):
    """Capture a snapshot of all model parameters as numpy arrays."""
    if policy._is_pytorch_model:
        return {
            name: param.data.cpu().numpy().copy()
            for name, param in policy._model.named_parameters()
        }
    else:
        from flax import nnx
        import jax
        _, state = nnx.split(policy._model)
        flat = jax.tree.leaves(state.to_pure_dict())
        return {str(i): np.asarray(v).copy() for i, v in enumerate(flat)}


def compute_param_diff(snap_before, snap_after):
    """Compare two parameter snapshots. Returns (num_changed, total, max_diff, mean_diff)."""
    total = 0
    changed = 0
    max_diff = 0.0
    sum_diff = 0.0
    n_elements = 0

    for key in snap_before:
        before = snap_before[key]
        after = snap_after[key]
        total += 1
        diff = np.abs(before.astype(np.float64) - after.astype(np.float64))
        if not np.allclose(before, after, atol=1e-7):
            changed += 1
        max_diff = max(max_diff, float(diff.max()))
        sum_diff += float(diff.sum())
        n_elements += diff.size

    mean_diff = sum_diff / n_elements if n_elements > 0 else 0.0
    return changed, total, max_diff, mean_diff


def test_dtype(args, dtype_name, mode="weights_only"):
    """Load model, apply quantization, verify weights changed."""
    logging.info(f"--- Testing dtype={dtype_name}, mode={mode} ---")

    # Load fresh policy
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    is_pytorch = policy._is_pytorch_model
    logging.info(f"  Model type: {'PyTorch' if is_pytorch else 'JAX/Flax'}")

    # Convert to float32 baseline
    if is_pytorch:
        policy._model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
    else:
        convert_model_to_float32_nnx(policy._model)

    # Snapshot before quantization
    snap_before = get_param_snapshot(policy)
    logging.info(f"  Captured {len(snap_before)} parameter groups before quantization")

    # Apply quantization
    t0 = time.time()
    if dtype_name == "float32":
        logging.info("  Skipping quantization (float32 baseline)")
    elif is_pytorch:
        quantize_model_weights(policy._model, dtype_name)
    else:
        quantize_model_weights_nnx(policy._model, dtype_name)
    quant_time = time.time() - t0

    # Snapshot after quantization
    snap_after = get_param_snapshot(policy)

    # Compare
    changed, total, max_diff, mean_diff = compute_param_diff(snap_before, snap_after)

    if dtype_name == "float32":
        if changed == 0:
            logging.info(f"  PASS  float32 baseline: no params changed (expected)")
            return True
        else:
            logging.error(f"  FAIL  float32 baseline: {changed}/{total} params changed unexpectedly")
            return False
    else:
        if changed > 0:
            logging.info(
                f"  PASS  {changed}/{total} param groups changed  "
                f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}  "
                f"quant_time={quant_time:.1f}s"
            )
            return True
        else:
            logging.error(f"  FAIL  No params changed for dtype={dtype_name}!")
            return False


def test_inference(args, dtype_name="float16", mode="weights_only", port=8099):
    """Load model, quantize, start server, make one inference call."""
    logging.info(f"--- Testing inference: dtype={dtype_name}, mode={mode}, port={port} ---")

    from openpi.serving import websocket_policy_server
    from openpi_client import websocket_client_policy
    import asyncio
    import threading

    # Load and quantize
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)
    is_pytorch = policy._is_pytorch_model

    if is_pytorch:
        policy._model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
    else:
        convert_model_to_float32_nnx(policy._model)

    if dtype_name != "float32":
        if is_pytorch:
            quantize_model_weights(policy._model, dtype_name)
        else:
            quantize_model_weights_nnx(policy._model, dtype_name)

    # Start server in background thread
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy.metadata,
    )

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Wait for server
    import urllib.request
    for _ in range(60):
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2)
            if resp.status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        logging.error("  FAIL  Server did not start within 60s")
        return False

    logging.info("  Server is ready, making test inference...")

    # Make a test inference
    try:
        client = websocket_client_policy.WebsocketClientPolicy("localhost", port)
        # Create fake observation matching DROID format
        fake_obs = {
            "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "pick up the cube",
        }
        result = client.infer(fake_obs)
        actions = result.get("actions")
        if actions is not None:
            actions = np.array(actions)
            has_nan = np.any(np.isnan(actions))
            has_inf = np.any(np.isinf(actions))
            logging.info(
                f"  Actions shape: {actions.shape}, "
                f"range: [{actions.min():.4f}, {actions.max():.4f}], "
                f"NaN: {has_nan}, Inf: {has_inf}"
            )
            if has_nan or has_inf:
                logging.error(f"  FAIL  Actions contain NaN/Inf!")
                return False
            logging.info(f"  PASS  Inference succeeded with dtype={dtype_name}")
            return True
        else:
            logging.error(f"  FAIL  No 'actions' in server response: {list(result.keys())}")
            return False
    except Exception as e:
        logging.error(f"  FAIL  Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test server quantization pipeline")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris",
                        help="Training config name")
    parser.add_argument("--checkpoint",
                        default="gs://openpi-assets/checkpoints/pi05_droid_jointpos",
                        help="Model checkpoint path")
    parser.add_argument("--dtypes", nargs="+", default=None,
                        help=f"Dtypes to test (default: all). Options: {' '.join(DTYPE_NAMES)}")
    parser.add_argument("--test-inference", action="store_true",
                        help="Also test serving and inference (slower)")
    parser.add_argument("--port", type=int, default=8099,
                        help="Port for inference test server")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    dtypes = args.dtypes or DTYPE_NAMES
    results = {}

    print()
    print("=" * 70)
    print("  SERVER QUANTIZATION INTEGRATION TEST")
    print("=" * 70)

    # Test each dtype (weight quantization only)
    for dtype_name in dtypes:
        try:
            passed = test_dtype(args, dtype_name)
            results[dtype_name] = passed
        except Exception as e:
            logging.error(f"  FAIL  {dtype_name}: {e}")
            import traceback
            traceback.print_exc()
            results[dtype_name] = False

    # Optional inference test
    if args.test_inference:
        print()
        # Test with float16 as a representative quantized dtype
        test_dt = "float16" if "float16" in dtypes else dtypes[0]
        try:
            inf_pass = test_inference(args, dtype_name=test_dt, port=args.port)
            results[f"inference_{test_dt}"] = inf_pass
        except Exception as e:
            logging.error(f"  FAIL  inference test: {e}")
            results[f"inference_{test_dt}"] = False

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<30} {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All tests passed!")
    else:
        print("  Some tests FAILED!")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
