#!/usr/bin/env python3
"""
FP Precision Sim-Eval Benchmark

Orchestrates running sim-evals across multiple FP quantization configurations.
For each (dtype, quantization mode) pair:
  1. Starts a quantized policy server
  2. Runs sim-eval for each scene (1, 2, 3) with N episodes
  3. Collects success rates
  4. Stops the server

Produces summary JSON and matplotlib graphs.

Usage:
    # Quick smoke test
    python run_benchmark.py --episodes 2 --scenes 1 --dtypes float32 float16 --modes weights_only

    # Full benchmark
    python run_benchmark.py --episodes 5 --scenes 1 2 3

    # Custom subset
    python run_benchmark.py --episodes 5 --scenes 1 2 3 --dtypes float32 float16 bfloat16 --modes weights_only both
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Paths (relative to this script)
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_DIR = SCRIPT_DIR / "openpi"
SIM_EVALS_DIR = SCRIPT_DIR / "sim-evals"
SERVE_SCRIPT = SCRIPT_DIR / "serve_quantized_policy.py"

ALL_DTYPES = [
    "float32", "float16", "bfloat16",
    "float8_e4m3", "float8_e5m2",
    "int32", "uint32", "boolean",
]

ALL_MODES = ["weights_only", "activations_only", "both"]

DTYPE_DISPLAY = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float8_e4m3": "fp8 e4m3",
    "float8_e5m2": "fp8 e5m2",
    "int32": "int32",
    "uint32": "uint32",
    "boolean": "boolean",
}

SCENE_COLORS = {
    1: "#2196F3",  # blue
    2: "#FF9800",  # orange
    3: "#4CAF50",  # green
}

SCENE_LABELS = {
    1: "Scene 1: Cube in Bowl",
    2: "Scene 2: Can in Mug",
    3: "Scene 3: Banana in Bin",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Server lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

def start_server(dtype, mode, port=8001, config="pi05_droid_jointpos_polaris",
                 checkpoint="gs://openpi-assets/checkpoints/pi05_droid_jointpos",
                 gpu=None, openpi_data_home=None, log_dir=None):
    """Start quantized policy server as a subprocess in the openpi environment."""
    cmd = [
        "uv", "run",
        str(SERVE_SCRIPT),
        "--config", config,
        "--checkpoint", checkpoint,
        "--dtype", dtype,
        "--mode", mode,
        "--port", str(port),
    ]
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
    }
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if openpi_data_home is not None:
        env["OPENPI_DATA_HOME"] = str(openpi_data_home)

    # Redirect stdout/stderr to log files to avoid pipe buffer deadlock.
    # If pipes fill up (64KB), the server process blocks on write() and hangs.
    stdout_file = None
    stderr_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"server_{dtype}_{mode}"
        stdout_file = open(log_dir / f"{log_name}_stdout.log", "w")
        stderr_file = open(log_dir / f"{log_name}_stderr.log", "w")

    proc = subprocess.Popen(
        cmd,
        cwd=str(OPENPI_DIR),
        env=env,
        stdout=stdout_file or subprocess.DEVNULL,
        stderr=stderr_file or subprocess.DEVNULL,
    )
    proc._log_files = (stdout_file, stderr_file)  # stash for cleanup
    return proc


def wait_for_server(port=8001, timeout=300, poll_interval=5):
    """Poll /healthz until server is ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(poll_interval)
    raise TimeoutError(f"Server did not become ready within {timeout}s")


def stop_server(proc):
    """Gracefully stop the server subprocess."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Close log files if any
    for f in getattr(proc, "_log_files", (None, None)):
        if f is not None:
            f.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Sim-eval execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim_eval(scene, episodes, port=8001):
    """Run sim-eval for a single scene. Returns the summary dict or None on failure."""
    cmd = [
        "uv", "run", "python", str(SIM_EVALS_DIR / "run_eval.py"),
        "--episodes", str(episodes),
        "--scene", str(scene),
        "--headless",
        "--port", str(port),
    ]

    # Don't set CUDA_VISIBLE_DEVICES for sim-eval: Isaac Sim uses Vulkan for
    # rendering (which ignores CUDA_VISIBLE_DEVICES) and conflicts with it.
    # Let Isaac Sim manage its own GPU selection.
    env = {**os.environ}
    env.pop("CUDA_VISIBLE_DEVICES", None)

    timeout_sec = episodes * 300  # 5 min per episode, generous

    # Record time before launching so we only pick up newly-created summaries
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SIM_EVALS_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            print(f"    [WARN] run_eval.py exited with code {result.returncode}")
            print(f"    stderr: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        print(f"    [ERROR] run_eval.py timed out after {timeout_sec}s")
        return None

    # Find summary.json created after we launched run_eval.py
    runs_dir = SIM_EVALS_DIR / "runs"
    summaries = [
        p for p in runs_dir.glob("*/*/summary.json")
        if p.stat().st_mtime >= start_time
    ]
    summaries.sort(key=lambda p: p.stat().st_mtime)
    if not summaries:
        print("    [ERROR] No summary.json found after run")
        return None

    latest = summaries[-1]
    run_dir = latest.parent
    with open(latest) as f:
        summary = json.load(f)

    # Collect video file paths relative to sim-evals/
    videos = sorted(run_dir.glob("episode_*.mp4"))
    summary["_run_dir"] = str(run_dir)
    summary["_videos"] = [str(v) for v in videos]

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Graph generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_graphs(results, scenes, output_dir):
    """Generate summary graphs from benchmark results."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    # Determine which modes were tested
    modes_tested = set()
    dtypes_tested = []
    for config_key, config_data in results.items():
        mode = config_data.get("mode", "none")
        dtype = config_data.get("dtype", "float32")
        modes_tested.add(mode)
        if dtype not in dtypes_tested:
            dtypes_tested.append(dtype)

    # ── Graph 1: Per-mode bar charts (grouped by scene) ──────────────────────
    mode_configs = [
        ("weights_only", "Weights Only", "#2196F3"),
        ("activations_only", "Activations Only", "#FF9800"),
        ("both", "Weights & Activations", "#F44336"),
    ]

    for mode_key, mode_title, _ in mode_configs:
        if mode_key not in modes_tested and "none" not in modes_tested:
            continue

        # Collect dtypes that have data for this mode
        mode_dtypes = []
        mode_data = {}  # {scene: [success_rates]}

        for s in scenes:
            mode_data[s] = []

        # Always include float32 baseline
        if "float32_none" in results:
            mode_dtypes.append("float32")
            for s in scenes:
                sr = results["float32_none"].get(f"scene_{s}", {}).get("success_rate", 0)
                mode_data[s].append(sr)

        for config_key, config_data in results.items():
            if config_key == "float32_none":
                continue
            if config_data.get("mode") != mode_key:
                continue
            dtype = config_data["dtype"]
            mode_dtypes.append(dtype)
            for s in scenes:
                sr = config_data.get(f"scene_{s}", {}).get("success_rate", 0)
                mode_data[s].append(sr)

        if len(mode_dtypes) == 0:
            continue

        x = np.arange(len(mode_dtypes))
        n_scenes = len(scenes)
        width = 0.8 / n_scenes

        fig, ax = plt.subplots(figsize=(max(10, len(mode_dtypes) * 1.5), 6))

        for i, s in enumerate(scenes):
            offset = (i - (n_scenes - 1) / 2) * width
            bars = ax.bar(x + offset, mode_data[s], width,
                          label=SCENE_LABELS.get(s, f"Scene {s}"),
                          color=SCENE_COLORS.get(s, "#999999"),
                          alpha=0.85, edgecolor="black", linewidth=0.5)

            # Value labels
            for bar, val in zip(bars, mode_data[s]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DTYPE_DISPLAY.get(d, d) for d in mode_dtypes], rotation=45, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Pi0 Sim-Eval — Success Rate by Precision — {mode_title}")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"success_rate_{mode_key}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved: success_rate_{mode_key}.png")

    # ── Graph 2: Overall summary (average success rate across scenes) ────────
    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.8), 6))

    config_keys = list(results.keys())
    avg_rates = []
    bar_labels = []
    bar_colors = []

    mode_color_map = {
        "none": "#9E9E9E",
        "weights_only": "#2196F3",
        "activations_only": "#FF9800",
        "both": "#F44336",
    }

    for config_key in config_keys:
        config_data = results[config_key]
        scene_rates = [
            config_data.get(f"scene_{s}", {}).get("success_rate", 0)
            for s in scenes
        ]
        avg = sum(scene_rates) / len(scene_rates) if scene_rates else 0
        avg_rates.append(avg)

        dtype = config_data.get("dtype", "?")
        mode = config_data.get("mode", "none")
        display_dtype = DTYPE_DISPLAY.get(dtype, dtype)
        if mode == "none":
            bar_labels.append(f"{display_dtype}\n(baseline)")
        else:
            bar_labels.append(f"{display_dtype}\n({mode.replace('_', ' ')})")
        bar_colors.append(mode_color_map.get(mode, "#999999"))

    x = np.arange(len(config_keys))
    bars = ax.bar(x, avg_rates, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, avg_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Average Success Rate")
    ax.set_title("Pi0 Sim-Eval — Average Success Rate by Quantization Config")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f"success_rate_overall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: success_rate_overall.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results, scenes):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS")
    print("=" * 80)

    scene_headers = "".join(f"  Scene {s:>2}" for s in scenes)
    header = f"{'Config':<30}{scene_headers}  {'Average':>8}"
    print(header)
    print("-" * len(header))

    for config_key, config_data in results.items():
        dtype = config_data.get("dtype", "?")
        mode = config_data.get("mode", "none")
        label = f"{dtype} ({mode})" if mode != "none" else f"{dtype} (baseline)"

        scene_vals = []
        for s in scenes:
            sr = config_data.get(f"scene_{s}", {}).get("success_rate", 0)
            scene_vals.append(sr)

        avg = sum(scene_vals) / len(scene_vals) if scene_vals else 0
        vals_str = "".join(f"  {v:>7.0%}" for v in scene_vals)
        print(f"{label:<30}{vals_str}  {avg:>7.0%}")

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="FP Precision Sim-Eval Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per scene (default: 5)")
    parser.add_argument("--scenes", type=int, nargs="+", default=[1, 2, 3],
                        help="Scene numbers to evaluate (default: 1 2 3)")
    parser.add_argument("--dtypes", nargs="+", default=ALL_DTYPES,
                        help=f"Dtype names to test (default: {' '.join(ALL_DTYPES)})")
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        help=f"Quantization modes (default: {' '.join(ALL_MODES)})")
    parser.add_argument("--port", type=int, default=8001,
                        help="Server port (default: 8001)")
    parser.add_argument("--server-timeout", type=int, default=300,
                        help="Server startup timeout in seconds (default: 300)")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris",
                        help="OpenPi training config name")
    parser.add_argument("--checkpoint",
                        default="gs://openpi-assets/checkpoints/pi05_droid_jointpos",
                        help="Model checkpoint path")
    parser.add_argument("--gpu", type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0' or '1')")
    parser.add_argument("--openpi-data-home", type=str, default=None,
                        help="OPENPI_DATA_HOME path for model cache")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = SCRIPT_DIR / "results" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  FP PRECISION SIM-EVAL BENCHMARK")
    print("=" * 80)
    print(f"  Episodes per scene: {args.episodes}")
    print(f"  Scenes: {args.scenes}")
    print(f"  Dtypes: {args.dtypes}")
    print(f"  Modes: {args.modes}")
    print(f"  GPU: {args.gpu or 'inherited from env'}")
    print(f"  OPENPI_DATA_HOME: {args.openpi_data_home or 'inherited from env'}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Build experiment configs
    configs = []

    # Float32 baseline (no quantization)
    if "float32" in args.dtypes:
        configs.append(("float32", "none"))

    # All other dtype × mode combinations
    for dtype in args.dtypes:
        if dtype == "float32":
            continue
        for mode in args.modes:
            configs.append((dtype, mode))

    total_configs = len(configs)
    total_runs = total_configs * len(args.scenes)
    est_minutes = total_runs * args.episodes * 2  # ~2 min per episode
    print(f"\n  Total configurations: {total_configs}")
    print(f"  Total evaluation runs: {total_runs}")
    print(f"  Estimated time: ~{est_minutes} minutes ({est_minutes / 60:.1f} hours)")
    print()

    results = {}
    video_index = []  # List of {dtype, mode, scene, episode, video_path, success}

    for ci, (dtype, mode) in enumerate(configs):
        config_key = f"{dtype}_{mode}"
        results[config_key] = {"dtype": dtype, "mode": mode}

        print(f"\n{'─' * 80}")
        print(f"  [{ci + 1}/{total_configs}] Config: {config_key}")
        print(f"{'─' * 80}")

        # Start server
        print(f"  Starting server (dtype={dtype}, mode={mode})...")
        server_proc = start_server(dtype, mode, port=args.port,
                                   config=args.config, checkpoint=args.checkpoint,
                                   gpu=args.gpu, openpi_data_home=args.openpi_data_home,
                                   log_dir=output_dir / "server_logs")

        try:
            print(f"  Waiting for server to be ready (timeout={args.server_timeout}s)...")
            wait_for_server(port=args.port, timeout=args.server_timeout)
            print(f"  Server is ready!")

            for si, scene in enumerate(args.scenes):
                print(f"\n  Scene {scene} ({si + 1}/{len(args.scenes)}), "
                      f"{args.episodes} episodes...")

                summary = run_sim_eval(scene, args.episodes, port=args.port)

                if summary:
                    sr = summary.get("success_rate", 0)
                    succ = summary.get("successful_episodes", 0)
                    total = summary.get("total_episodes", 0)
                    results[config_key][f"scene_{scene}"] = {
                        "success_rate": sr,
                        "successful_episodes": succ,
                        "total_episodes": total,
                    }
                    status = "PASS" if sr > 0 else "FAIL"
                    print(f"    Result: {succ}/{total} ({sr:.0%}) [{status}]")

                    # Collect video index entries
                    run_dir = Path(summary.get("_run_dir", ""))
                    for video_path in summary.get("_videos", []):
                        vp = Path(video_path)
                        # Extract episode number from filename (episode_0.mp4 -> 0)
                        ep_str = vp.stem.replace("episode_", "")
                        ep_num = int(ep_str) if ep_str.isdigit() else -1

                        # Try to read per-episode success from state log
                        ep_success = None
                        state_log = run_dir / "state_logs" / f"episode_{ep_num}_state.json"
                        if state_log.exists():
                            with open(state_log) as sf:
                                ep_success = json.load(sf).get("success", None)

                        video_index.append({
                            "dtype": dtype,
                            "mode": mode,
                            "scene": scene,
                            "episode": ep_num,
                            "success": ep_success,
                            "video_path": str(vp),
                        })
                else:
                    results[config_key][f"scene_{scene}"] = {
                        "success_rate": 0,
                        "successful_episodes": 0,
                        "total_episodes": args.episodes,
                        "error": "run_eval failed",
                    }
                    print(f"    Result: ERROR - run_eval failed")

        except TimeoutError as e:
            print(f"  [ERROR] {e}")
            for scene in args.scenes:
                results[config_key][f"scene_{scene}"] = {
                    "success_rate": 0,
                    "successful_episodes": 0,
                    "total_episodes": args.episodes,
                    "error": "server timeout",
                }

        finally:
            print(f"  Stopping server...")
            stop_server(server_proc)
            # Wait for GPU memory to be freed
            time.sleep(10)

        # Compute average success rate for this config
        scene_rates = [
            results[config_key].get(f"scene_{s}", {}).get("success_rate", 0)
            for s in args.scenes
        ]
        results[config_key]["avg_success_rate"] = (
            sum(scene_rates) / len(scene_rates) if scene_rates else 0
        )

    # ── Save results ─────────────────────────────────────────────────────────
    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "episodes_per_scene": args.episodes,
            "scenes": args.scenes,
            "dtypes": args.dtypes,
            "modes": args.modes,
        },
        "results": results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    video_index_path = output_dir / "video_index.json"
    with open(video_index_path, "w") as f:
        json.dump(video_index, f, indent=2)
    print(f"  Video index saved to: {video_index_path} ({len(video_index)} entries)")

    # ── Generate graphs ──────────────────────────────────────────────────────
    print("\n  Generating graphs...")
    generate_graphs(results, args.scenes, output_dir)

    # ── Print summary ────────────────────────────────────────────────────────
    print_summary_table(results, args.scenes)

    print(f"\n  All outputs saved to: {output_dir}")
    print("  Done!")


if __name__ == "__main__":
    main()
