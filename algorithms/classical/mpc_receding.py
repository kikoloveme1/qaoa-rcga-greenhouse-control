"""
Proper Receding Horizon MPC for Greenhouse Climate Control.

Uses SLSQP with hard bounds and linear rate constraints.
Unlike the block-iterative L-BFGS-B approach, this is a standard MPC:
  - At each hour t, optimizes over prediction horizon N_p
  - Applies only the first control
  - Recedes and repeats
"""
import sys, os
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from time import perf_counter
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig


@dataclass
class MPCResult:
    method: str = "MPC-Receding"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)


def run_mpc_receding(env, seed=42, pred_horizon=6, max_iter=200, init_mode="random", verbose=False):
    """
    Standard receding horizon MPC.

    Parameters
    ----------
    env : GreenhouseEnv
    seed : int
    pred_horizon : int
        Prediction horizon in hours (N_p). Default 6h.
    max_iter : int
        Max iterations for SLSQP per step.

    Returns
    -------
    MPCResult
    """
    rng = np.random.default_rng(seed)
    cfg = env.config
    n_steps = cfg.T_steps  # 24

    # Per-hour variable bounds (T, L, C, H)
    bounds_low, bounds_high = env.bounds()
    per_low = bounds_low[:4]
    per_high = bounds_high[:4]

    # Max rate of change per variable
    # Handle PerturbationEnv which wraps base_config
    if hasattr(cfg, 'base_config'):
        bc = cfg.base_config
        max_delta = np.array([bc.max_dT, bc.max_dL, bc.max_dC, bc.max_dH])
    else:
        max_delta = np.array([cfg.max_dT, cfg.max_dL, cfg.max_dC, cfg.max_dH])

    t0 = perf_counter()
    total_evals = 0

    # Initialize plan
    if init_mode == "midpoint":
        plan = np.tile((per_low + per_high) / 2, (n_steps, 1))
    else:  # random (default)
        plan = np.zeros((n_steps, 4))
        for v in range(4):
            plan[:, v] = rng.uniform(per_low[v], per_high[v], n_steps)
        # Enforce rate constraints on initial plan
        for t in range(1, n_steps):
            for v in range(4):
                lo = plan[t-1, v] - max_delta[v]
                hi = plan[t-1, v] + max_delta[v]
                plan[t, v] = np.clip(plan[t, v], lo, hi)

    for t in range(n_steps):
        # Determine optimization window
        opt_start = t
        opt_end = min(t + pred_horizon, n_steps)
        n_opt = opt_end - opt_start

        if n_opt == 0:
            break

        # Decision variables: plan[opt_start:opt_end].flatten() -> 4 * n_opt dims
        x0 = plan[opt_start:opt_end].flatten()

        # Bounds for this window
        bounds_list = []
        for _ in range(n_opt):
            bounds_list.extend([
                (per_low[0], per_high[0]),
                (per_low[1], per_high[1]),
                (per_low[2], per_high[2]),
                (per_low[3], per_high[3]),
            ])
        bounds = Bounds(
            [b[0] for b in bounds_list],
            [b[1] for b in bounds_list],
        )

        # Rate constraints: |x_{i+1} - x_i| <= max_delta for adjacent hours in window
        # Also need constraint for transition from t-1 to t (if t > 0)
        A_rows = []
        A_lb = []
        A_ub = []

        # Internal rate constraints within the window
        for i in range(n_opt - 1):
            for v in range(4):
                row = np.zeros(4 * n_opt)
                row[i * 4 + v] = -1.0
                row[(i + 1) * 4 + v] = 1.0
                A_rows.append(row)
                A_lb.append(-max_delta[v])
                A_ub.append(max_delta[v])

        # Transition constraint from t-1 to t (if t > 0)
        if t > 0:
            # plan[t-1] is fixed, plan[t] is x0[0:4]
            prev_controls = plan[t - 1]
            for v in range(4):
                row = np.zeros(4 * n_opt)
                row[v] = 1.0  # coefficient for plan[t, v]
                A_rows.append(row)
                A_lb.append(prev_controls[v] - max_delta[v])
                A_ub.append(prev_controls[v] + max_delta[v])

        constraints = []
        if A_rows:
            A_matrix = np.array(A_rows)
            constraints.append(LinearConstraint(
                A_matrix, np.array(A_lb), np.array(A_ub)
            ))

        # Objective function
        n_evals_inner = [0]

        def objective(x_flat):
            n_evals_inner[0] += 1
            x = x_flat.reshape(n_opt, 4)
            # Create full 24h plan: fixed early hours + optimized window + placeholder later
            full_plan = plan.copy()
            full_plan[opt_start:opt_end] = x
            # Later hours: use midpoint (won't be evaluated yet)
            if opt_end < n_steps:
                full_plan[opt_end:] = np.tile((per_low + per_high) / 2, (n_steps - opt_end, 1))
            f, _details = env.fitness(full_plan)
            return -float(f)

        try:
            res = minimize(
                objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter, "ftol": 1e-6},
            )
            x_opt = res.x.reshape(n_opt, 4)
            plan[opt_start:opt_end] = x_opt
            total_evals += n_evals_inner[0]
            if verbose and t % 4 == 0:
                f_check, d_check = env.fitness(plan)
                print(f"  t={t:2d}: profit={d_check['profit_sgd']:.2f}, feasible={d_check['is_feasible']}, evals={n_evals_inner[0]}")
        except Exception as e:
            if verbose:
                print(f"  t={t}: SLSQP failed ({e}), using previous plan")
            total_evals += n_evals_inner[0]

    # Final evaluation
    final_f, final_d = env.fitness(plan)
    total_evals += 1
    elapsed = perf_counter() - t0

    return MPCResult(
        method="MPC-Receding",
        best_fitness=float(final_f),
        best_yield=float(final_d["total_yield"]),
        best_energy=float(final_d["total_energy"]),
        best_penalty=float(final_d["total_penalty"])
        + float(final_d.get("perturbation_total_penalty", 0.0)),
        feasible=final_d["is_feasible"],
        n_evals=total_evals,
        elapsed=elapsed,
        best_x=plan,
    )


# Self-test
if __name__ == "__main__":
    print("=== Proper Receding Horizon MPC - Quick Test ===")
    env = GreenhouseEnv(seed=42)

    for horizon in [4, 6, 8]:
        print(f"\nPrediction horizon N_p={horizon}:")
        result = run_mpc_receding(env, seed=42, pred_horizon=horizon, max_iter=100, verbose=True)
        print(f"  Result: profit={result.best_fitness:.2f}, yield={result.best_yield:.1f}, "
              f"energy={result.best_energy:.1f}, feasible={result.feasible}, "
              f"evals={result.n_evals}, time={result.elapsed:.1f}s")
