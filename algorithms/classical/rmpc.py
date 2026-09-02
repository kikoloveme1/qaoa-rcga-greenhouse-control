# -*- coding: utf-8 -*-
"""Fixed-margin MPC with a boundary-proximity penalty.

Uses rolling-window SLSQP. Tube RMPC is defined in tube_rmpc.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from time import perf_counter
from dataclasses import dataclass, field
from typing import Optional

from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig


def _environment_objective(fitness_value, details):
    """Use the environment's returned objective, including scenario costs."""

    del details
    return float(fitness_value)

@dataclass
class TightenedMPCResult:
    method: str = "TightenedMPC"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)


def run_tightened_mpc(env, seed=42, pred_horizon=4,
                      sigma_T=1.5, sigma_solar=80.0, sigma_RH=5.0,
                      safety_factor=2.0, uncertainty_penalty_weight=1.0,
                      max_iter=150, init_mode="random", verbose=False):
    """Heuristic MPC with constraint tightening and a proximity penalty.

    Parameters
    ----------
    env : GreenhouseEnv or PerturbationEnv
    seed : int
    pred_horizon : int - prediction horizon (hours)
    sigma_T : float - expected outdoor temperature uncertainty (C)
    sigma_solar : float - expected solar radiation uncertainty (W/m2)
    sigma_RH : float - expected humidity uncertainty (%)
    safety_factor : float - multiplier for constraint tightening
    uncertainty_penalty_weight : float - weight on uncertainty penalty
    max_iter : int - SLSQP max iterations
    verbose : bool

    Returns
    -------
    TightenedMPCResult
    """
    rng = np.random.default_rng(seed)
    cfg = env.config
    n_steps = cfg.T_steps
    if hasattr(cfg, 'base_config'):
        bc = cfg.base_config
    else:
        bc = cfg

    # Tube-based constraint tightening
    # The idea: since outdoor T can vary by +/-sigma_T, the indoor temperature
    # control must be more conservative. We tighten bounds by safety_factor*sigma.
    T_margin = safety_factor * sigma_T
    H_margin = safety_factor * sigma_RH
    C_margin = safety_factor * (sigma_solar * 0.05)  # CO2 varies with ventilation (solar proxy)

    per_low = np.array([bc.T_lower + T_margin, 0.0, bc.C_lower + 50.0, bc.H_lower + H_margin])
    per_high = np.array([bc.T_upper - T_margin, bc.L_upper * 0.75, bc.C_upper - 100.0, bc.H_upper - H_margin])
    # Ensure bounds are still valid
    per_low = np.maximum(per_low, [bc.T_lower, 0.0, bc.C_lower, bc.H_lower])
    per_high = np.minimum(per_high, [bc.T_upper, bc.L_upper, bc.C_upper, bc.H_upper])
    mid = (per_low + per_high) / 2

    max_delta = np.array([bc.max_dT, bc.max_dL, bc.max_dC, bc.max_dH])

    t0 = perf_counter()
    total_evals = 0

    # Initialize plan
    # Random initial plan (seed-dependent)
    plan = np.zeros((n_steps, 4))
    for v in range(4):
        plan[:, v] = rng.uniform(per_low[v], per_high[v], n_steps)
    # Enforce rate constraints on initial plan
    for t in range(1, n_steps):
        for v in range(4):
            lo = plan[t-1, v] - max_delta[v]
            hi = plan[t-1, v] + max_delta[v]
            plan[t, v] = np.clip(plan[t, v], lo, hi)

    # Random initialization is now the default (seed-dependent)
    # To use midpoint: set init_mode="midpoint"

    for t in range(n_steps):
        opt_start = t
        opt_end = min(t + pred_horizon, n_steps)
        n_opt = opt_end - opt_start
        if n_opt == 0:
            break

        n_vars = n_opt * 4
        x0 = plan[opt_start:opt_end].flatten()

        # Bounds
        bl = []; bu = []
        for _ in range(n_opt):
            bl.extend([per_low[v] for v in range(4)])
            bu.extend([per_high[v] for v in range(4)])
        bounds = Bounds(bl, bu)

        # Rate constraints
        Ar, Alb, Aub = [], [], []
        for i in range(n_opt - 1):
            for v in range(4):
                row = np.zeros(n_vars)
                row[i*4+v] = -1.0
                row[(i+1)*4+v] = 1.0
                Ar.append(row)
                Alb.append(-max_delta[v])
                Aub.append(max_delta[v])
        if t > 0:
            prev = plan[t-1]
            for v in range(4):
                row = np.zeros(n_vars)
                row[v] = 1.0
                Ar.append(row)
                Alb.append(prev[v] - max_delta[v])
                Aub.append(prev[v] + max_delta[v])

        constraints = [LinearConstraint(np.array(Ar), np.array(Alb), np.array(Aub))] if Ar else []

        n_evals_inner = [0]

        def robust_objective(x_flat):
            """Robust MPC objective: nominal profit minus uncertainty penalty."""
            n_evals_inner[0] += 1
            x = x_flat.reshape(n_opt, 4)
            full_plan = plan.copy()
            full_plan[opt_start:opt_end] = x
            if opt_end < n_steps:
                full_plan[opt_end:] = np.tile(mid, (n_steps - opt_end, 1))

            # Nominal evaluation
            f_val, details = env.fitness(full_plan)
            nom_profit = _environment_objective(f_val, details)

            # Uncertainty penalty: penalize controls that are too close to tightened bounds
            # The more aggressive the control, the higher the risk under disturbance
            unc_penalty = 0.0
            for ti in range(n_opt):
                hour = opt_start + ti
                # T penalty: being near T bounds is risky
                T_sp = x[ti, 0]
                to_low = bc.T_lower + T_margin * 0.5
                to_high = bc.T_upper - T_margin * 0.5
                if T_sp < to_low:
                    unc_penalty += (to_low - T_sp) ** 2 * 0.5
                if T_sp > to_high:
                    unc_penalty += (T_sp - to_high) ** 2 * 0.5

                # CO2 penalty: low CO2 is risky under ventilation uncertainty
                C_inj = x[ti, 2]
                if C_inj < 400:
                    unc_penalty += (400 - C_inj) ** 2 * 0.01

                # H penalty
                H_sp = x[ti, 3]
                if H_sp < bc.H_lower + H_margin * 1.5:
                    unc_penalty += ((bc.H_lower + H_margin * 1.5) - H_sp) ** 2 * 0.02
                if H_sp > bc.H_upper - H_margin * 1.5:
                    unc_penalty += (H_sp - (bc.H_upper - H_margin * 1.5)) ** 2 * 0.02

            return -(nom_profit - uncertainty_penalty_weight * unc_penalty)

        try:
            res = minimize(
                robust_objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": max_iter, "ftol": 1e-6},
            )
            if res.success:
                x_opt = res.x.reshape(n_opt, 4)
                plan[opt_start:opt_end] = x_opt
        except Exception:
            pass

        total_evals += n_evals_inner[0]

        if verbose and t % 4 == 0:
            f_check, d_check = env.fitness(plan)
            print(f"  t={t:2d}: profit={d_check.get('profit_sgd',0):+.2f}, feas={d_check.get('is_feasible',False)}, evals_acc={total_evals}")

    # Final evaluation
    final_f, final_d = env.fitness(plan)
    total_evals += 1
    pert_pen = final_d.get("perturbation_total_penalty", 0.0)
    adj_fitness = float(final_d.get("profit_sgd", final_f)) - pert_pen

    elapsed = perf_counter() - t0

    return TightenedMPCResult(
        method="TightenedMPC",
        best_fitness=adj_fitness,
        best_yield=float(final_d.get("total_yield", 0)),
        best_energy=float(final_d.get("total_energy", 0)),
        best_penalty=float(final_d.get("total_penalty", 0)) + pert_pen,
        feasible=final_d.get("is_feasible", False),
        n_evals=total_evals,
        elapsed=elapsed,
        best_x=plan,
    )


# Self-test
if __name__ == "__main__":
    print("=== Tightened MPC Quick Test ===\n")
    env = GreenhouseEnv(seed=42)

    for sf in [1.0, 2.0, 3.0]:
        print(f"Safety factor={sf}:")
        r = run_tightened_mpc(env, seed=42, pred_horizon=4, safety_factor=sf, max_iter=100, verbose=True)
        print(f"  profit={r.best_fitness:.2f}, yield={r.best_yield:.1f}, energy={r.best_energy:.1f}, "
              f"feasible={r.feasible}, evals={r.n_evals}, time={r.elapsed:.1f}s\n")
