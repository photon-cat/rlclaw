"""
CMA-ES training script for the MLP controller.

Usage:
    cd /home/jacob/rlclaw && python3 -m src.algos.cmaes_train --num_segs 20 --max_time 300
"""
import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "commaai"))

import cma

from vendor.commaai.tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from src.controllers.cmaes_mlp import Controller, TOTAL_PARAMS


# Paths
MODEL_PATH = str(PROJECT_ROOT / "vendor" / "commaai" / "models" / "tinyphysics.onnx")
DATA_DIR = PROJECT_ROOT / "vendor" / "commaai" / "data"
SAVE_DIR = PROJECT_ROOT / "src" / "algos" / "data"
SAVE_PATH = SAVE_DIR / "cmaes_best.npy"


def evaluate_single(args):
    """Evaluate a single candidate on a single segment. Returns total_cost."""
    params, data_path = args
    model = TinyPhysicsModel(MODEL_PATH, debug=False)
    controller = Controller.from_params(params)
    sim = TinyPhysicsSimulator(model, str(data_path), controller=controller, debug=False)
    cost = sim.rollout()
    return cost['total_cost']


def evaluate_candidate(params, data_files, pool):
    """Evaluate a candidate across all segments, return mean total_cost."""
    work = [(params, f) for f in data_files]
    costs = pool.map(evaluate_single, work)
    return np.mean(costs)


def main():
    parser = argparse.ArgumentParser(description="CMA-ES training for MLP controller")
    parser.add_argument("--num_segs", type=int, default=20, help="Number of segments to evaluate on")
    parser.add_argument("--max_time", type=float, default=300, help="Max training time in seconds")
    parser.add_argument("--pop_size", type=int, default=40, help="CMA-ES population size")
    parser.add_argument("--sigma0", type=float, default=0.5, help="Initial step size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Resume from saved params")
    args = parser.parse_args()

    np.random.seed(args.seed)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Select data files
    all_files = sorted(DATA_DIR.iterdir())
    data_files = all_files[:args.num_segs]
    print(f"Training on {len(data_files)} segments")
    print(f"Total params: {TOTAL_PARAMS}")
    print(f"Population size: {args.pop_size}")
    print(f"Max time: {args.max_time}s")

    # Initial solution
    if args.resume and SAVE_PATH.exists():
        x0 = np.load(SAVE_PATH)
        print(f"Resumed from {SAVE_PATH}")
    else:
        x0 = np.zeros(TOTAL_PARAMS)

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['seed'] = args.seed
    opts['popsize'] = args.pop_size
    opts['timeout'] = args.max_time
    opts['verbose'] = -1  # suppress cma's own output, we print our own

    es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)

    best_cost = float('inf')
    best_params = x0.copy()
    gen = 0
    start_time = time.time()

    pool = Pool(processes=args.workers)

    try:
        while not es.stop():
            elapsed = time.time() - start_time
            if elapsed > args.max_time:
                print(f"Time limit reached ({elapsed:.1f}s)")
                break

            candidates = es.ask()

            # Evaluate all candidates in parallel
            # Flatten: each candidate evaluated on all segments
            all_work = []
            for c in candidates:
                for f in data_files:
                    all_work.append((c, f))

            all_costs = pool.map(evaluate_single, all_work)

            # Reshape and average per candidate
            n_cand = len(candidates)
            n_segs = len(data_files)
            cost_matrix = np.array(all_costs).reshape(n_cand, n_segs)
            fitness = cost_matrix.mean(axis=1)

            es.tell(candidates, fitness.tolist())

            # Track best
            gen_best_idx = np.argmin(fitness)
            gen_best_cost = fitness[gen_best_idx]
            gen_mean_cost = np.mean(fitness)

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_params = candidates[gen_best_idx].copy()
                np.save(SAVE_PATH, best_params)

            elapsed = time.time() - start_time
            print(f"Gen {gen:>3d} | best: {gen_best_cost:>8.3f} | mean: {gen_mean_cost:>8.3f} | "
                  f"overall_best: {best_cost:>8.3f} | time: {elapsed:>6.1f}s")

            gen += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pool.terminate()
        pool.join()

    # Save final best
    np.save(SAVE_PATH, best_params)
    print(f"\nTraining complete after {gen} generations, {time.time() - start_time:.1f}s")
    print(f"Best cost: {best_cost:.3f}")
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
