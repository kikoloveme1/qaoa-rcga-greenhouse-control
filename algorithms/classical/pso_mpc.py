# -*- coding: utf-8 -*-
"""Particle-swarm MPC with rolling-window plan warm starts.

Each step optimizes a control window and retains its first action.
Reference: Gong et al. (2023), PSO-MPC for greenhouse control."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from time import perf_counter
from dataclasses import dataclass, field
from typing import Optional

from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig


def _environment_objective(fitness_value, details):
    """Return the environment objective including scenario costs."""

    del details
    return float(fitness_value)

@dataclass
class PSOMPCResult:
    method: str = "PSO-MPC"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)


def run_pso_mpc(env, seed=42, pred_horizon=6, swarm_size=40, pso_iters=60, init_mode="random",
                w=0.729, c1=1.494, c2=1.494, vs_max=0.15, verbose=False):
    """PSO-MPC: PSO as the optimizer in receding horizon MPC.

    Parameters
    ----------
    env : GreenhouseEnv or PerturbationEnv
    seed : int
    pred_horizon : int - N_p prediction horizon (hours)
    swarm_size : int - PSO swarm size per step
    pso_iters : int - PSO iterations per step
    w : float - inertia weight
    c1, c2 : float - cognitive/social coefficients
    vs_max : float - max velocity as fraction of bound range
    verbose : bool

    Returns
    -------
    PSOMPCResult
    """
    rng = np.random.default_rng(seed)
    cfg = env.config
    n_steps = cfg.T_steps
    # Check if we are on a PerturbationEnv
    if hasattr(cfg, 'base_config'):
        bc = cfg.base_config
    else:
        bc = cfg

    bounds_low, bounds_high = env.bounds()
    per_low = bounds_low[:4]
    per_high = bounds_high[:4]
    mid = (per_low + per_high) / 2
    max_delta = np.array([bc.max_dT, bc.max_dL, bc.max_dC, bc.max_dH])
    bound_range = per_high - per_low

    t0 = perf_counter()
    total_evals = 0

    # Initialize full plan
    if init_mode == "midpoint":
        plan = np.tile(mid, (n_steps, 1))
    else:  # random (default)
        plan = np.zeros((n_steps, 4))
        for v in range(4):
            plan[:, v] = rng.uniform(per_low[v], per_high[v], n_steps)
        # Enforce rate constraints
        for t in range(1, n_steps):
            for v in range(4):
                lo = plan[t-1, v] - max_delta[v]
                hi = plan[t-1, v] + max_delta[v]
                plan[t, v] = np.clip(plan[t, v], lo, hi)

    for t in range(n_steps):
        opt_start = t
        opt_end = min(t + pred_horizon, n_steps)
        n_opt = opt_end - opt_start
        if n_opt == 0:
            break

        n_vars = n_opt * 4

        # Initialize swarm
        swarm_pos = np.zeros((swarm_size, n_vars))
        swarm_vel = np.zeros((swarm_size, n_vars))
        swarm_best_pos = np.zeros((swarm_size, n_vars))
        swarm_best_fit = np.full(swarm_size, -np.inf)

        for i in range(swarm_size):
            # Use current plan as center with perturbation
            center = plan[opt_start:opt_end].flatten()
            swarm_pos[i] = center + rng.normal(0, 0.1, n_vars) * np.tile(bound_range, n_opt)
            for dim in range(n_vars):
                v_idx = dim % 4
                lb = per_low[v_idx]
                ub = per_high[v_idx]
                swarm_pos[i, dim] = np.clip(swarm_pos[i, dim], lb, ub)
            swarm_vel[i] = rng.uniform(-0.05, 0.05, n_vars) * np.tile(bound_range, n_opt)

        # Evaluate initial swarm
        for i in range(swarm_size):
            x_opt = swarm_pos[i].reshape(n_opt, 4)
            full_plan = plan.copy()
            full_plan[opt_start:opt_end] = x_opt
            if opt_end < n_steps:
                full_plan[opt_end:] = np.tile(mid, (n_steps - opt_end, 1))
            f_val, details = env.fitness(full_plan)
            total_evals += 1
            profit = _environment_objective(f_val, details)
            # Penalize rate constraint violations
            pen = _rate_penalty(full_plan[:opt_end], max_delta)
            adj_profit = profit - pen * 5.0
            swarm_best_pos[i] = swarm_pos[i].copy()
            swarm_best_fit[i] = adj_profit

        global_best_idx = np.argmax(swarm_best_fit)
        global_best_pos = swarm_best_pos[global_best_idx].copy()
        global_best_fit = swarm_best_fit[global_best_idx]

        # PSO iterations
        for it in range(pso_iters):
            for i in range(swarm_size):
                r1 = rng.random(n_vars)
                r2 = rng.random(n_vars)
                swarm_vel[i] = (w * swarm_vel[i]
                    + c1 * r1 * (swarm_best_pos[i] - swarm_pos[i])
                    + c2 * r2 * (global_best_pos - swarm_pos[i]))
                # Clamp velocity
                vmax = vs_max * np.tile(bound_range, n_opt)
                swarm_vel[i] = np.clip(swarm_vel[i], -vmax, vmax)
                swarm_pos[i] = swarm_pos[i] + swarm_vel[i]
                # Clamp position
                for dim in range(n_vars):
                    v_idx = dim % 4
                    swarm_pos[i, dim] = np.clip(swarm_pos[i, dim], per_low[v_idx], per_high[v_idx])

                # Evaluate
                x_opt = swarm_pos[i].reshape(n_opt, 4)
                full_plan = plan.copy()
                full_plan[opt_start:opt_end] = x_opt
                if opt_end < n_steps:
                    full_plan[opt_end:] = np.tile(mid, (n_steps - opt_end, 1))
                f_val, details = env.fitness(full_plan)
                total_evals += 1
                profit = _environment_objective(f_val, details)
                pen = _rate_penalty(full_plan[:opt_end], max_delta)
                adj_profit = profit - pen * 5.0

                if adj_profit > swarm_best_fit[i]:
                    swarm_best_fit[i] = adj_profit
                    swarm_best_pos[i] = swarm_pos[i].copy()
                    if adj_profit > global_best_fit:
                        global_best_fit = adj_profit
                        global_best_pos = swarm_pos[i].copy()

        # Apply first control from optimized window
        x_opt = global_best_pos.reshape(n_opt, 4)
        plan[t] = x_opt[0].copy()

        # Also fill rest of window for warm-start
        plan[opt_start:opt_end] = x_opt

        if verbose and t % 4 == 0:
            f_check, d_check = env.fitness(plan)
            print(f"  t={t:2d}: profit={d_check.get('profit_sgd',0):+.2f}, feas={d_check.get('is_feasible',False)}, evals_acc={total_evals}")

    # Final evaluation
    final_f, final_d = env.fitness(plan)
    total_evals += 1

    # Account for perturbation penalty
    pert_pen = final_d.get("perturbation_total_penalty", 0.0)
    adj_fitness = float(final_f)

    elapsed = perf_counter() - t0

    return PSOMPCResult(
        method="PSO-MPC",
        best_fitness=adj_fitness,
        best_yield=float(final_d.get("total_yield", 0)),
        best_energy=float(final_d.get("total_energy", 0)),
        best_penalty=float(final_d.get("total_penalty", 0)) + pert_pen,
        feasible=final_d.get("is_feasible", False),
        n_evals=total_evals,
        elapsed=elapsed,
        best_x=plan,
    )


def _rate_penalty(plan_2d, max_delta):
    """Compute sum of rate-of-change violations."""
    pen = 0.0
    for t in range(1, plan_2d.shape[0]):
        for v in range(4):
            delta = abs(plan_2d[t, v] - plan_2d[t-1, v])
            if delta > max_delta[v]:
                pen += (delta - max_delta[v]) ** 2
    return pen


# Self-test
if __name__ == "__main__":
    print("=== PSO-MPC Quick Test ===\\n")
    env = GreenhouseEnv(seed=42)

    for hp in [4, 6]:
        print(f"Prediction horizon N_p={hp}:")
        r = run_pso_mpc(env, seed=42, pred_horizon=hp, swarm_size=30, pso_iters=40, verbose=True)
        print(f"  profit={r.best_fitness:.2f}, yield={r.best_yield:.1f}, energy={r.best_energy:.1f}, "
              f"feasible={r.feasible}, evals={r.n_evals}, time={r.elapsed:.1f}s")
