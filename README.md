# FP Precision Sim-Eval Benchmark

Measures the real-world impact of floating-point quantization on the Pi0 robot policy by running simulated evaluations at different precisions and measuring task success rate.

<!-- This complements [fp-benchmarks](https://github.com/emersonhsieh/fp-benchmarks), which measures RMSE of quantized model outputs. Here we measure whether quantization actually causes the robot to fail tasks. -->

## Scenes

| Scene | Task | Target | Container |
|-------|------|--------|-----------|
| 1 | Put cube in bowl | Rubik's cube | Bowl |
| 2 | Put can in mug | Potted meat can | Mug |
| 3 | Put banana in bin | Banana | KLT bin |

## Precision Types

| Type | Bits | Method |
|------|------|--------|
| float32 | 32 | Baseline (no quantization) |
| float16 | 16 | Cast round-trip |
| bfloat16 | 16 | Cast round-trip |
| float8_e4m3 | 8 | Cast with range clamping |
| float8_e5m2 | 8 | Cast with range clamping |
| int32 | 32 | Min-max linear quantization |
| uint32 | 32 | Min-max linear quantization |
| boolean | 1 | Median-threshold binarization |

## Quantization Modes

| Mode | Description |
|------|-------------|
| weights_only | Quantize model parameters before inference |
| activations_only | Forward hooks quantize intermediate layer outputs |
| both | Weights and activations quantized simultaneously |

## Installation

### Prerequisites

- NVIDIA GPU with ~40GB VRAM (A100 recommended)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:emersonhsieh/fp-sim-benchmark.git
cd fp-sim-benchmark
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Set up openpi (policy server)

```bash
cd openpi
uv venv --python 3.11 .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Patch transformers for openpi compatibility (see note below)
cp -r src/openpi/models_pytorch/transformers_replace/* \
  .venv/lib/python3.11/site-packages/transformers/

# Download simulation assets
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets

cd ..
```

To verify openpi is working, start the policy server manually:

```bash
# in ./openpi
export CUDA_VISIBLE_DEVICES=1
export OPENPI_DATA_HOME=/scratch/emersonhsieh/data/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
uv run ./scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

**Why the transformers patch is needed:** OpenPi's Pi0 model uses a custom PaliGemma architecture (Gemma 2B language model + SigLIP vision encoder + action expert) that differs from the standard HuggingFace implementations. The `transformers_replace/` directory contains modified versions of the `gemma`, `paligemma`, and `siglip` model files that are compatible with OpenPi's architecture. These patched files are copied directly into the installed `transformers` package, replacing the stock model implementations. The Pi0 model checks at initialization that this patch has been applied correctly and will refuse to load without it.

### 3. Set up sim-evals (simulation environment)

```bash
cd sim-evals
```

Follow the setup instructions in [sim-evals/README.md](sim-evals/README.md). This requires IsaacLab / Isaac Sim.

```bash
# Download simulation assets
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

Ensure `run_eval.py` works standalone before running benchmarks (start the policy server from step 2 first in a separate terminal):

```bash
# Test that sim-evals works
uv run python run_eval.py --episodes 1 --scene 1 --port 8001 --headless
```

```bash
cd ..
```

### 4. Install benchmark dependencies

The orchestrator script (`run_benchmark.py`) only needs matplotlib and numpy:

```bash
pip install matplotlib numpy
```

### 5. Environment variables

The benchmark script handles these automatically, but they are important to understand:

| Variable | Purpose | Default |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Select which GPU to use | Inherited from shell |
| `OPENPI_DATA_HOME` | Where openpi caches model weights | Inherited from shell |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory for JAX/XLA | `0.5` (set by benchmark) |

If you have multiple GPUs, use `--gpu` to select one. If you need a custom model cache location, use `--openpi-data-home`. Example:

```bash
python run_benchmark.py --gpu 0 --openpi-data-home /data/openpi_models --episodes 2 --scenes 1
```

## How It Works

The quantization is applied **on the fly inside the policy server process**:

1. `run_benchmark.py` launches `serve_quantized_policy.py` via `uv run` from the `openpi/` directory (with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` and optional `CUDA_VISIBLE_DEVICES` / `OPENPI_DATA_HOME`)
2. The server loads the Pi0 model in float32
3. It applies the requested quantization (weight casting, activation hooks, or both) **in-process before serving**
4. By the time the WebSocket server starts listening, the model is already quantized
5. `run_eval.py` is launched via `uv run` from the `sim-evals/` directory -- it connects to the server and runs episodes as usual, unaware of quantization
6. After evaluation, the server is stopped, and a new one is started for the next quantization config

This means openpi and sim-evals don't need any code changes. The quantization is injected by `serve_quantized_policy.py` which monkey-patches `torch.compile` and applies quantization between model loading and serving.

## Usage

### Quick smoke test

Run float32 and float16 with weights-only quantization on scene 1, 2 episodes:

```bash
python run_benchmark.py \
    --episodes 2 \
    --scenes 1 \
    --dtypes float32 float16 \
    --modes weights_only
```

### Run all float types only

Test all floating-point precisions with all quantization modes, 5 episodes per scene:

```bash
python run_benchmark.py \
    --episodes 5 \
    --scenes 1 2 3 \
    --dtypes float32 float16 bfloat16 float8_e4m3 float8_e5m2 \
    --modes weights_only activations_only both
```

### Run all types with all quantization modes (full benchmark)

Test every precision type (floats, integers, boolean) with all three quantization modes:

```bash
python run_benchmark.py \
    --episodes 5 \
    --scenes 1 2 3 \
    --dtypes float32 float16 bfloat16 float8_e4m3 float8_e5m2 int32 uint32 boolean \
    --modes weights_only activations_only both
```

This is equivalent to the default (no flags needed):

```bash
python run_benchmark.py --episodes 5 --scenes 1 2 3
```

### Custom model checkpoint

```bash
python run_benchmark.py \
    --config pi05_droid_jointpos_polaris \
    --checkpoint gs://openpi-assets/checkpoints/pi05_droid_jointpos \
    --episodes 5
```

### Multi-GPU machine

Select a specific GPU and set a model cache directory:

```bash
python run_benchmark.py \
    --gpu 0 \
    --openpi-data-home /data/openpi_models \
    --episodes 5 \
    --scenes 1 2 3
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--episodes` | 5 | Episodes per scene |
| `--scenes` | 1 2 3 | Scene numbers to evaluate |
| `--dtypes` | all 8 types | Precision types to test |
| `--modes` | all 3 modes | Quantization modes |
| `--port` | 8001 | WebSocket server port |
| `--server-timeout` | 300 | Server startup timeout (seconds) |
| `--config` | pi05_droid_jointpos_polaris | OpenPi training config name |
| `--checkpoint` | gs://openpi-assets/checkpoints/pi05_droid_jointpos | Model checkpoint path |
| `--gpu` | inherited | CUDA_VISIBLE_DEVICES value (e.g. `0`) |
| `--openpi-data-home` | inherited | OPENPI_DATA_HOME path for model cache |

## Output

Results are saved to `results/<timestamp>/`:

```
results/2026-02-19_16-30-00/
  results.json                    # All success rates
  success_rate_weights_only.png   # Bar chart: weights-only mode
  success_rate_activations_only.png
  success_rate_both.png
  success_rate_overall.png        # Summary across all configs
```

## Time Estimates

| Configuration | Configs | Runs | Estimated Time |
|---------------|---------|------|---------------|
| Smoke test (2 dtypes, 1 mode, 1 scene, 2 ep) | 2 | 2 | ~10 min |
| Float types only (5 dtypes, 3 modes, 3 scenes, 5 ep) | 13 | 39 | ~4 hours |
| Full benchmark (8 dtypes, 3 modes, 3 scenes, 5 ep) | 22 | 66 | ~11 hours |

Each episode takes roughly 2 minutes. Server restart adds ~1-2 minutes per configuration.

## Architecture

```
fp-sim-benchmark/
  run_benchmark.py              # Orchestrator
  serve_quantized_policy.py     # Quantized policy server
  quantization_utils.py         # Shared quantization functions
  openpi/                       # Git submodule (policy model + server)
  sim-evals/                    # Git submodule (Isaac Sim environments)
  results/                      # Output (created at runtime)
```

```
run_benchmark.py (orchestrator)
    |
    |-- For each (dtype, mode) config:
    |       |
    |       |-- Start: serve_quantized_policy.py (in openpi/ env via uv run)
    |       |     Loads model -> float32 -> applies quantization -> WebSocket server
    |       |
    |       |-- Wait: polls /healthz endpoint until ready
    |       |
    |       |-- Run: sim-evals/run_eval.py (in sim-evals/ env)
    |       |     Connects to server -> runs episodes -> writes summary.json
    |       |
    |       |-- Collect: reads summary.json, records success rate
    |       |
    |       |-- Stop: terminates server, waits for GPU memory cleanup
    |
    |-- Aggregate results -> results.json
    |-- Generate graphs -> *.png
```
