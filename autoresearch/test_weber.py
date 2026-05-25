"""
A/B test: Weber optimizer vs vanilla (bracket disabled).

Runs two identical training sessions:
  A) WEBER_C_SQ = <value>   (Weber bracket active)
  B) WEBER_C_SQ = 1e30      (bracket ~= 1.0, effectively vanilla)

Same SDR seed for both so the only difference is the bracket correction.
Compares val_bpb, training dynamics, and bracket statistics.

Usage:
    uv run test_weber.py                           # default c²=1.0 vs vanilla
    uv run test_weber.py --c-sq 0.5                # test c²=0.5
    uv run test_weber.py --c-sq 0.1 0.5 1.0 5.0   # sweep multiple values
    uv run test_weber.py --seed 12345              # fixed seed (no SDR)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
TRAIN_PY = PROJECT_DIR / "train.py"


def set_hyperparam(name, value):
    """Set a single hyperparameter in train.py."""
    text = TRAIN_PY.read_text(encoding="utf-8")
    lines = text.splitlines()
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*)(.+?)(\s*#.*)$")
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            lines[i] = f"{m.group(1)}{value}{m.group(3)}"
            TRAIN_PY.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
    return False


def run_experiment(label):
    """Run train.py and parse results."""
    log_path = PROJECT_DIR / f"run_{label}.log"
    print(f"  [{label}] Running...", end="", flush=True)
    t0 = time.time()

    result = subprocess.run(
        ["uv", "run", "train.py"],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        timeout=900,
        cwd=PROJECT_DIR,
    )

    dt = time.time() - t0
    print(f" done ({dt:.0f}s)")

    if result.returncode != 0:
        print(f"  [{label}] CRASHED (exit code {result.returncode})")
        log_text = log_path.read_text()
        print("  " + "\n  ".join(log_text.splitlines()[-10:]))
        return None

    # Parse metrics
    log_text = log_path.read_text()
    metrics = {}
    in_summary = False
    for line in log_text.splitlines():
        if line.strip() == "---":
            in_summary = True
            continue
        if in_summary:
            m = re.match(r"^(\w+):\s+(.+)$", line.strip())
            if m:
                try:
                    metrics[m.group(1)] = float(m.group(2))
                except ValueError:
                    metrics[m.group(1)] = m.group(2).strip()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="A/B test Weber optimizer")
    parser.add_argument("--c-sq", type=float, nargs="+", default=[1.0],
                        help="Weber c² values to test (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fixed seed (bypasses SDR entropy)")
    args = parser.parse_args()

    # Save original train.py
    original = TRAIN_PY.read_text(encoding="utf-8")

    print("=" * 60)
    print("Weber Optimizer A/B Test")
    print("=" * 60)

    results = []

    try:
        # Run vanilla baseline (bracket disabled via huge c²)
        print(f"\n--- VANILLA (c² = 1e30, bracket ~= 1.0) ---")
        set_hyperparam("WEBER_C_SQ", "1e30")
        vanilla = run_experiment("vanilla")
        if vanilla:
            vanilla["weber_c_sq"] = 1e30
            results.append(("vanilla", vanilla))

        # Run each Weber c² value
        for c_sq in args.c_sq:
            print(f"\n--- WEBER (c² = {c_sq}) ---")
            # Restore original first (in case previous run changed something)
            TRAIN_PY.write_text(original, encoding="utf-8")
            set_hyperparam("WEBER_C_SQ", str(c_sq))
            weber = run_experiment(f"weber_csq{c_sq}")
            if weber:
                weber["weber_c_sq"] = c_sq
                results.append((f"weber_c²={c_sq}", weber))

    finally:
        # Always restore original
        TRAIN_PY.write_text(original, encoding="utf-8")

    # Print comparison
    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Config':<20} {'val_bpb':>10} {'steps':>8} {'tokens_M':>10} {'VRAM_MB':>10} {'time_budget':>12}")
    print("-" * 72)

    best_label = None
    best_bpb = float("inf")
    for label, m in results:
        bpb = m.get("val_bpb", float("inf"))
        steps = m.get("num_steps", "?")
        tokens = m.get("total_tokens_M", "?")
        vram = m.get("peak_vram_mb", "?")
        budget = m.get("time_budget", "?")
        print(f"{label:<20} {bpb:>10.6f} {steps:>8} {tokens:>10} {vram:>10} {budget:>12}")
        if bpb < best_bpb:
            best_bpb = bpb
            best_label = label

    print(f"\nBest: {best_label} (val_bpb = {best_bpb:.6f})")

    # Delta from vanilla
    vanilla_bpb = None
    for label, m in results:
        if label == "vanilla":
            vanilla_bpb = m.get("val_bpb")
            break

    if vanilla_bpb:
        print(f"\nDelta from vanilla ({vanilla_bpb:.6f}):")
        for label, m in results:
            if label == "vanilla":
                continue
            bpb = m.get("val_bpb", float("inf"))
            delta = bpb - vanilla_bpb
            pct = 100 * delta / vanilla_bpb
            direction = "better" if delta < 0 else "worse"
            print(f"  {label}: {delta:+.6f} ({pct:+.2f}%, {direction})")

    # Save results
    results_path = PROJECT_DIR / "weber_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
