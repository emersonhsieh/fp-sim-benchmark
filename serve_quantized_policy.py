#!/usr/bin/env python3
"""
Quantized Policy Server for FP Precision Benchmarking.

Loads an OpenPi policy, optionally applies quantization (weights, activations, or both),
and serves via WebSocket. Designed to be run inside the openpi uv environment.

Usage (from the openpi directory):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run \
        /path/to/serve_quantized_policy.py \
        --config pi05_droid_jointpos_polaris \
        --checkpoint gs://openpi-assets/checkpoints/pi05_droid_jointpos \
        --dtype float16 \
        --mode weights_only \
        --port 8001
"""

import sys
import os
import argparse
import logging
import socket

# Add this script's directory to path for quantization_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Monkey-patch torch.compile before importing openpi models (same as fp-benchmarks).
# This prevents issues with quantized tensors in compiled graphs.
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
from openpi.serving import websocket_policy_server

from quantization_utils import (
    quantize_model_weights, ActivationQuantizer,
    quantize_model_weights_nnx,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a quantized OpenPi policy")
    parser.add_argument("--config", required=True, help="Training config name (e.g. pi05_droid_jointpos_polaris)")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or S3 path")
    parser.add_argument("--dtype", default="float32", help="Target dtype for quantization")
    parser.add_argument("--mode", default="none",
                        choices=["none", "weights_only", "activations_only", "both"],
                        help="Quantization mode")
    parser.add_argument("--port", type=int, default=8001, help="WebSocket server port")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    logging.info(f"Configuration: dtype={args.dtype}, mode={args.mode}, port={args.port}")

    # 1. Load the policy (same as serve_policy.py)
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)

    # 2. For PyTorch models, convert to float32 baseline before quantization.
    #    For JAX/Flax models, skip this step to avoid GPU OOM — quantize_numpy
    #    handles bfloat16→float32 conversion on CPU internally.
    is_pytorch = policy._is_pytorch_model
    if is_pytorch:
        logging.info("Converting PyTorch model to float32 baseline...")
        policy._model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")

    # 3. Apply quantization
    act_quantizer = None

    if args.dtype != "float32" and args.mode != "none":
        if args.mode in ("weights_only", "both"):
            logging.info(f"Quantizing weights to {args.dtype}...")
            if is_pytorch:
                quantize_model_weights(policy._model, args.dtype)
            else:
                quantize_model_weights_nnx(policy._model, args.dtype)
            logging.info("Weight quantization complete.")

        if args.mode in ("activations_only", "both"):
            if is_pytorch:
                logging.info(f"Attaching activation quantizer for {args.dtype}...")
                act_quantizer = ActivationQuantizer(policy._model, args.dtype)
                logging.info(f"Activation quantizer attached ({len(act_quantizer.hooks)} hooks).")
            else:
                logging.warning(
                    "Activation quantization is not supported for JAX/Flax models. "
                    "Skipping activation quantization. Only weight quantization will be applied."
                )

    # 4. Serve
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Serving quantized policy on {local_ip}:{args.port}")
    logging.info(f"  dtype={args.dtype}, mode={args.mode}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
