"""Run the complete principal benchmark across CPU processes with resume support."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_protocol.protocol import Protocol
from paper_protocol.runner import build_landscape, run_one, summarize, write_json


_PROTOCOL = None
_LANDSCAPE = None
_OUTPUT = None
_DATA = None


def _initialize_worker(protocol, landscape, output, data_path):
    global _PROTOCOL, _LANDSCAPE, _OUTPUT, _DATA
    _PROTOCOL = protocol
    _LANDSCAPE = landscape
    _OUTPUT = output
    _DATA = data_path


def _run_task(task):
    method, scenario, seed = task
    result = run_one(
        _PROTOCOL,
        method,
        scenario,
        seed,
        _OUTPUT,
        _LANDSCAPE,
        data_path=_DATA,
    )
    return {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "profit": result["profit"],
        "penalty": result["penalty"],
        "seconds": result["total_seconds"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fitness-workers", type=int, default=2)
    parser.add_argument("--backend", choices=("aer", "numpy"), default="aer")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.fitness_workers < 1:
        parser.error("worker counts must be positive")

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "QAOA_AER_THREADS",
        "SAC_PPO_TORCH_THREADS",
    ):
        os.environ[name] = "1"

    protocol = replace(
        Protocol(),
        backend=args.backend,
        fitness_workers=args.fitness_workers,
    )
    methods = (
        "qaoa_rcga",
        "rcga",
        "pso_mpc",
        "tube_rmpc",
        "mpc_receding",
        "sac_ppo",
    )
    tasks = [
        (method, scenario, seed)
        for scenario in protocol.scenarios
        for seed in protocol.seeds
        for method in methods
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "protocol.json"
    expected = json.loads(json.dumps(asdict(protocol)))
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != expected:
        parser.error(f"protocol mismatch in {args.out}; choose a fresh output directory")
    write_json(manifest, expected)

    started = perf_counter()
    landscape = build_landscape(protocol, args.out)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(protocol, landscape, args.out, args.data),
    ) as pool:
        futures = {pool.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            print(
                f"[{completed:03d}/{len(tasks)}] {row['method']}/{row['scenario']}/{row['seed']} "
                f"profit={row['profit']:.6f} penalty={row['penalty']:.3g} "
                f"run={row['seconds']:.1f}s wall={perf_counter()-started:.1f}s",
                flush=True,
            )
    report = summarize(args.out)
    print(
        json.dumps(
            {
                "runs": len(tasks),
                "wall_seconds": perf_counter() - started,
                "groups": len(report["groups"]),
                "output": str(args.out.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
