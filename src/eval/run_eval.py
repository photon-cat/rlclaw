#!/usr/bin/env python3
"""
Evaluation script for the comma.ai Controls Challenge v2.

Runs a controller from src/controllers/ through the tinyphysics simulator
on N segments and reports lataccel_cost, jerk_cost, and total_cost.

Usage:
    python src/eval/run_eval.py --controller mpc --num_segs 100
    python src/eval/run_eval.py --controller mpc --num_segs 5 --max_workers 2
"""

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map

# Add vendor/commaai to path so we can import tinyphysics internals
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_DIR = REPO_ROOT / "vendor" / "commaai"
sys.path.insert(0, str(VENDOR_DIR))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

MODEL_PATH = str(VENDOR_DIR / "models" / "tinyphysics.onnx")
DATA_DIR = VENDOR_DIR / "data"
RESULTS_PATH = REPO_ROOT / "src" / "eval" / "results.json"


def run_single_segment(data_path, controller_module, model_path):
    """Run a single segment rollout. Designed to be called via process_map."""
    model = TinyPhysicsModel(model_path, debug=False)
    mod = importlib.import_module(controller_module)
    controller = mod.Controller()
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    return cost


def main():
    parser = argparse.ArgumentParser(description="Evaluate a controller on the controls challenge")
    parser.add_argument("--controller", type=str, required=True,
                        help="Controller name in src/controllers/ (e.g., 'mpc', 'pid')")
    parser.add_argument("--num_segs", type=int, default=100,
                        help="Number of segments to evaluate (default: 100)")
    parser.add_argument("--max_workers", type=int, default=16,
                        help="Max parallel workers (default: 16)")
    parser.add_argument("--save", action="store_true",
                        help="Save results to results.json")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag/name for this experiment in results.json")
    args = parser.parse_args()

    # Resolve controller module path
    controller_module = f"src.controllers.{args.controller}"

    # Verify controller can be imported
    src_dir = str(REPO_ROOT)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        mod = importlib.import_module(controller_module)
        assert hasattr(mod, "Controller"), f"Module {controller_module} has no Controller class"
    except Exception as e:
        print(f"ERROR: Could not import {controller_module}: {e}")
        sys.exit(1)

    # Get data files
    files = sorted(DATA_DIR.iterdir())[:args.num_segs]
    print(f"Evaluating controller '{args.controller}' on {len(files)} segments")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Workers: {args.max_workers}")
    print()

    # Run evaluation
    t0 = time.time()
    run_fn = partial(run_single_segment,
                     controller_module=controller_module,
                     model_path=MODEL_PATH)
    chunksize = max(1, len(files) // (args.max_workers * 4))
    results = process_map(run_fn, files, max_workers=args.max_workers, chunksize=chunksize)
    elapsed = time.time() - t0

    # Aggregate
    costs_df = pd.DataFrame(results)
    avg_lataccel = float(np.mean(costs_df["lataccel_cost"]))
    avg_jerk = float(np.mean(costs_df["jerk_cost"]))
    avg_total = float(np.mean(costs_df["total_cost"]))

    print(f"\n{'='*60}")
    print(f"Controller: {args.controller}")
    print(f"Segments:   {len(files)}")
    print(f"Time:       {elapsed:.1f}s ({elapsed/len(files):.2f}s/seg)")
    print(f"{'='*60}")
    print(f"  lataccel_cost:  {avg_lataccel:.4f}")
    print(f"  jerk_cost:      {avg_jerk:.4f}")
    print(f"  total_cost:     {avg_total:.4f}")
    print(f"{'='*60}")

    # Percentile breakdown
    print(f"\nTotal cost percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(costs_df["total_cost"], p)
        print(f"  p{p}: {val:.2f}")

    # Save results
    if args.save:
        tag = args.tag or args.controller
        result_entry = {
            "controller": args.controller,
            "tag": tag,
            "num_segs": len(files),
            "total_cost": round(avg_total, 4),
            "lataccel_cost": round(avg_lataccel, 4),
            "jerk_cost": round(avg_jerk, 4),
            "elapsed_s": round(elapsed, 1),
        }

        if RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                data = json.load(f)
        else:
            data = {"experiments": [], "baselines": {}}

        data["experiments"].append(result_entry)
        with open(RESULTS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
