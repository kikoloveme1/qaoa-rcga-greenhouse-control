# -*- coding: utf-8 -*-
"""Evolution-strategy policy search with symmetric perturbations and step clipping.

State: [sin(hour), cos(hour), T_out/40, solar/1000, price/0.3].
Actions: normalized temperature, light, CO2 injection and humidity controls."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from time import perf_counter
from dataclasses import dataclass, field
from typing import Optional

from environment.greenhouse_model import GreenhouseEnv, GreenhouseConfig


@dataclass
class ESPolicySearchResult:
    method: str = "ESPolicySearch"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)


def _forward_net(x, W1, b1, W2, b2, W3, b3):
    """Forward pass through 3-layer tanh network."""
    h = np.tanh(x @ W1 + b1)
    h = np.tanh(h @ W2 + b2)
    return h @ W3 + b3


def _get_state(t, env):
    """Construct normalized state vector. Works with PerturbationEnv."""
    state_env = env.eval_env if hasattr(env, 'eval_env') else env
    return np.array([
        np.sin(2 * np.pi * t / 24),
        np.cos(2 * np.pi * t / 24),
        state_env._T_out[t] / 40.0,
        state_env._solar[t] / 1000.0,
        state_env._tou_prices[t] / 0.3,
    ])


def _generate_plan(params, env, per_low, per_high, max_delta, deterministic=True):
    """Generate a 24h control plan from policy parameters."""
    W1, b1, W2, b2, W3, b3, log_std = params
    n_steps = env.n_steps
    rng = np.random.default_rng()
    plan = np.zeros((n_steps, 4))
    actions = []  # Store raw actions for log prob
    for t in range(n_steps):
        state = _get_state(t, env)
        mean = _forward_net(state, W1, b1, W2, b2, W3, b3)
        if deterministic:
            raw = mean
        else:
            std = np.exp(np.clip(log_std, -20, 2))
            raw = mean + rng.normal(0, 1, 4) * std
        action = np.clip(np.tanh(raw), -0.999, 0.999)
        actions.append((mean, raw, action))
        scaled = per_low + (action + 1) / 2 * (per_high - per_low)
        plan[t] = np.clip(scaled, per_low, per_high)
    # Enforce rate constraints
    for t in range(1, n_steps):
        for v in range(4):
            lo = plan[t-1, v] - max_delta[v]
            hi = plan[t-1, v] + max_delta[v]
            plan[t, v] = np.clip(plan[t, v], lo, hi)
    return plan, actions


def _flatten_params(params):
    """Flatten all network parameters to a single vector."""
    W1, b1, W2, b2, W3, b3, log_std = params
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3, log_std])


def _unflatten_params(flat, s_dim=5, h_dim=16, a_dim=4):
    """Unflatten parameter vector back to tuple."""
    idx = 0
    s1 = s_dim * h_dim; W1 = flat[idx:idx+s1].reshape(s_dim, h_dim); idx += s1
    s2 = h_dim; b1 = flat[idx:idx+s2]; idx += s2
    s3 = h_dim * h_dim; W2 = flat[idx:idx+s3].reshape(h_dim, h_dim); idx += s3
    s4 = h_dim; b2 = flat[idx:idx+s4]; idx += s4
    s5 = h_dim * a_dim; W3 = flat[idx:idx+s5].reshape(h_dim, a_dim); idx += s5
    s6 = a_dim; b3 = flat[idx:idx+s6]; idx += s6
    log_std = flat[idx:idx+a_dim]
    return (W1, b1, W2, b2, W3, b3, log_std)


def _init_params(rng, s_dim=5, h_dim=16, a_dim=4, scale=0.1):
    """Initialize network parameters."""
    W1 = rng.normal(0, scale, (s_dim, h_dim))
    b1 = np.zeros(h_dim)
    W2 = rng.normal(0, scale / np.sqrt(h_dim), (h_dim, h_dim))
    b2 = np.zeros(h_dim)
    W3 = rng.normal(0, scale / np.sqrt(h_dim), (h_dim, a_dim))
    b3 = np.zeros(a_dim)
    log_std = np.ones(a_dim) * np.log(0.3)
    return (W1, b1, W2, b2, W3, b3, log_std)


def run_es_policy_search(env, seed=42, budget=20000, hidden_dim=16,
                         lr=0.01, gamma=0.98, clip_eps=0.2,
                         entropy_coef=0.01, verbose=False):
    """Run the ES policy-search baseline.

    Uses a gradient-free ES-style perturbation approach for efficiency.

    Parameters
    ----------
    env : GreenhouseEnv or PerturbationEnv
    seed : int
    budget : int - max environment evaluations
    hidden_dim : int - hidden layer size
    lr : float - learning rate
    gamma : float - discount factor
    clip_eps : float - PPO clipping parameter
    entropy_coef : float - entropy bonus coefficient
    verbose : bool

    Returns
    -------
    ESPolicySearchResult
    """
    rng = np.random.default_rng(seed)
    n_steps = env.n_steps
    full_low, full_high = env.bounds()
    per_low = full_low[:4]
    per_high = full_high[:4]

    if hasattr(env.config, 'base_config'):
        bc = env.config.base_config
    else:
        bc = env.config
    max_delta = np.array([bc.max_dT, bc.max_dL, bc.max_dC, bc.max_dH])

    t0 = perf_counter()

    # Initialize policy
    params = _init_params(rng, s_dim=5, h_dim=hidden_dim, a_dim=4)
    params_old = _init_params(rng, s_dim=5, h_dim=hidden_dim, a_dim=4)
    flat = _flatten_params(params)
    flat_old = flat.copy()

    best_profit = -np.inf
    best_plan = None
    best_feasible = False
    best_yield = 0.0
    best_energy = 0.0
    best_penalty = 0.0
    n_evals = 0
    convergence = []

    C = 4  # Effective population size for ES-style gradient
    sigma_pert = 0.05  # Perturbation noise std

    while n_evals < budget:
        # Generate plans with current policy (deterministic for evaluation)
        plan, _ = _generate_plan(params, env, per_low, per_high, max_delta, deterministic=True)
        f_val, details = env.fitness(plan)
        n_evals += 1
        profit = float(details.get("profit_sgd", f_val))
        convergence.append(profit)

        if profit > best_profit:
            best_profit = profit
            best_plan = plan.copy()
            best_feasible = details.get("is_feasible", False)
            best_yield = float(details.get("total_yield", 0))
            best_energy = float(details.get("total_energy", 0))
            best_penalty = float(details.get("total_penalty", 0))

        # ES-style gradient estimation: evaluate C perturbations
        perturbations = []
        profits_plus = []
        profits_minus = []

        for c in range(C):
            if n_evals >= budget:
                break
            eps = rng.normal(0, sigma_pert, flat.shape)
            # Plus perturbation
            flat_plus = flat + eps
            params_plus = _unflatten_params(flat_plus)
            plan_p, _ = _generate_plan(params_plus, env, per_low, per_high, max_delta, deterministic=True)
            _, details_p = env.fitness(plan_p)
            n_evals += 1
            profit_p = float(details_p.get("profit_sgd", 0))
            if profit_p > best_profit:
                best_profit = profit_p
                best_plan = plan_p.copy()
                best_feasible = details_p.get("is_feasible", False)
                best_yield = float(details_p.get("total_yield", 0))
                best_energy = float(details_p.get("total_energy", 0))
                best_penalty = float(details_p.get("total_penalty", 0))

            # Minus perturbation
            flat_minus = flat - eps
            params_minus = _unflatten_params(flat_minus)
            plan_m, _ = _generate_plan(params_minus, env, per_low, per_high, max_delta, deterministic=True)
            _, details_m = env.fitness(plan_m)
            n_evals += 1
            profit_m = float(details_m.get("profit_sgd", 0))
            if profit_m > best_profit:
                best_profit = profit_m
                best_plan = plan_m.copy()
                best_feasible = details_m.get("is_feasible", False)
                best_yield = float(details_m.get("total_yield", 0))
                best_energy = float(details_m.get("total_energy", 0))
                best_penalty = float(details_m.get("total_penalty", 0))

            perturbations.append(eps)
            profits_plus.append(profit_p)
            profits_minus.append(profit_m)

            if n_evals >= budget:
                break

        if len(perturbations) == 0:
            break

        # Compute ES gradient: (1/C) * sum over c of (f_plus - f_minus) * eps / (2 * sigma^2)
        grad_es = np.zeros_like(flat)
        for i in range(len(perturbations)):
            fd = profits_plus[i] - profits_minus[i]
            grad_es += fd * perturbations[i]
        grad_es /= (len(perturbations) * 2 * sigma_pert**2)

        # Clip the parameter update to the trust region
        # For ES, we approximate the policy ratio using a simple trust-region
        # The "ratio" is 1 at the center, and we clip the gradient update

        # Normalize gradient
        g_norm = np.sqrt(np.sum(grad_es**2))
        if g_norm > 1e-8:
            grad_es = grad_es / g_norm

        # Update with PPO-style clipping on change magnitude
        delta = lr * grad_es
        # Clip per-coordinate change
        delta = np.clip(delta, -clip_eps * 0.1, clip_eps * 0.1)

        # Add entropy bonus by adding noise to log_std component
        log_std_idx = len(flat) - 4
        delta[log_std_idx:log_std_idx+4] += entropy_coef * 0.001

        flat_new = flat + delta
        flat_old = flat.copy()
        flat = flat_new
        params = _unflatten_params(flat)

        if verbose and n_evals % 2000 < (2*C + 1):
            print(f"  ES-policy evals={n_evals}/{budget}, best={best_profit:.2f}, grad_norm={g_norm:.4f}")

    elapsed = perf_counter() - t0

    # Account for perturbation penalty
    if best_plan is not None:
        _, final_d = env.fitness(best_plan)
        pert_pen = final_d.get("perturbation_total_penalty", 0.0)
        adj_profit = best_profit - pert_pen
    else:
        adj_profit = best_profit
        best_plan = np.zeros((n_steps, 4))

    return ESPolicySearchResult(
        method="ESPolicySearch",
        best_fitness=adj_profit,
        best_yield=best_yield,
        best_energy=best_energy,
        best_penalty=best_penalty,
        feasible=best_feasible,
        n_evals=n_evals,
        elapsed=elapsed,
        best_x=best_plan,
        convergence=convergence,
    )


# Self-test
if __name__ == "__main__":
    print("=== ES Policy Search Quick Test ===\n")
    env = GreenhouseEnv(seed=42)

    r = run_es_policy_search(env, seed=42, budget=5000, verbose=True)
    print(f"  profit={r.best_fitness:.2f}, yield={r.best_yield:.1f}, energy={r.best_energy:.1f}, "
          f"penalty={r.best_penalty:.1f}, feasible={r.feasible}, evals={r.n_evals}, time={r.elapsed:.1f}s")
