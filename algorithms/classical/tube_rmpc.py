"""Tube-based robust model predictive control for greenhouse operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

from environment.greenhouse_model import GREENHOUSE_VOLUME, GreenhouseEnv


@dataclass(frozen=True)
class TubeRMPCConfig:
    prediction_horizon: int = 6
    max_iter: int = 300
    finite_difference: float = 1e-4
    state_cost: float = 1.0
    control_cost: float = 0.1
    disturbance_radius: tuple = (0.5, 20.0, 2.0, 0.01, 1.0, 0.005)


@dataclass
class TubeRMPCResult:
    method: str = "TubeRMPC"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)
    feedback_gain: Optional[np.ndarray] = None
    tube_radii: Optional[np.ndarray] = None


def propagate_box(closed_loop, disturbance_radius, steps):
    """Propagate a centred interval tube under linear closed-loop dynamics."""
    matrix = np.asarray(closed_loop, dtype=float)
    disturbance = np.asarray(disturbance_radius, dtype=float)
    if matrix.shape != (disturbance.size, disturbance.size):
        raise ValueError("closed_loop and disturbance dimensions do not match")
    if steps < 1 or np.any(disturbance < 0):
        raise ValueError("steps must be positive and disturbance radii nonnegative")
    radius = np.zeros_like(disturbance)
    propagated = []
    for _ in range(steps):
        radius = np.abs(matrix) @ radius + disturbance
        propagated.append(radius.copy())
    return np.asarray(propagated)


def tighten_box(low, high, radius):
    """Compute the Pontryagin difference of two axis-aligned boxes."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    radius = np.asarray(radius, dtype=float)
    tightened_low = low + radius
    tightened_high = high - radius
    if np.any(tightened_low > tightened_high):
        raise ValueError("tube is larger than the admissible constraint box")
    return tightened_low, tightened_high


def _initial_state(model):
    cfg = model.config
    return np.array(
        [
            cfg.T_out_mean,
            600.0,
            float(model._RH_out[0]),
            cfg.LAI_initial,
            cfg.DM_initial,
            cfg.SWC_initial,
        ],
        dtype=float,
    )


def greenhouse_transition(model, state, control, hour):
    """Advance the greenhouse state used by the robust feedback model."""
    cfg = model.config
    t = min(int(hour), cfg.T_steps - 1)
    temperature, co2, humidity, lai, dry_matter, swc = np.asarray(state, dtype=float)
    t_sp, light, co2_injection, rh_sp = np.asarray(control, dtype=float)
    outdoor_temperature = float(model._T_out[t])
    solar = float(model._solar[t])
    par = solar * 0.5 + light

    temporary_rate = model.photosynthesis_rate(t_sp, par, co2, humidity)
    co2_next = model.co2_balance(
        co2_injection * model._co2_injection_scale[t],
        co2,
        t_sp,
        outdoor_temperature,
        temporary_rate,
        lai,
    )
    vpd = model.vpd(t_sp, humidity)
    transpiration = model.transpiration_rate(par, vpd, lai)
    absolute_humidity = model.rh_to_ah(humidity, temperature)
    absolute_humidity += transpiration / (GREENHOUSE_VOLUME * 1000.0)
    ventilation = model.ventilation_rate(t_sp, outdoor_temperature)
    outdoor_ah = model.rh_to_ah(float(model._RH_out[t]), outdoor_temperature)
    absolute_humidity -= ventilation * (absolute_humidity - outdoor_ah)
    uncontrolled_rh = float(model.ah_to_rh(absolute_humidity, t_sp))
    humidity_next = np.clip(
        humidity + cfg.humidity_tracking * (rh_sp - humidity) + uncontrolled_rh - humidity,
        0.0,
        100.0,
    )
    actual_transpiration = model.transpiration_rate(
        par, model.vpd(t_sp, humidity_next), lai
    )
    irrigation = model.irrigation_demand(actual_transpiration, swc)
    swc_next = swc + irrigation / (cfg.soil_depth * 1000.0)
    swc_next -= (actual_transpiration / 1000.0) / (cfg.soil_depth * 1000.0)
    water_factor = model.water_stress_response(swc_next)
    production = model.photosynthesis_rate(t_sp, par, co2_next, humidity_next) * water_factor
    lai_next, dry_matter_next, _ = model.update_crop_state(
        production, t_sp, lai, dry_matter
    )
    return np.array(
        [t_sp, co2_next, humidity_next, lai_next, dry_matter_next, swc_next],
        dtype=float,
    )


def linearize_transition(model, state, control, hour, relative_step=1e-4):
    """Finite-difference linearization around one state-control pair."""
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    n_state, n_control = state.size, control.size
    a = np.empty((n_state, n_state), dtype=float)
    b = np.empty((n_state, n_control), dtype=float)
    for i in range(n_state):
        step = relative_step * max(abs(state[i]), 1.0)
        plus, minus = state.copy(), state.copy()
        plus[i] += step
        minus[i] -= step
        a[:, i] = (
            greenhouse_transition(model, plus, control, hour)
            - greenhouse_transition(model, minus, control, hour)
        ) / (2.0 * step)
    for i in range(n_control):
        step = relative_step * max(abs(control[i]), 1.0)
        plus, minus = control.copy(), control.copy()
        plus[i] += step
        minus[i] -= step
        b[:, i] = (
            greenhouse_transition(model, state, plus, hour)
            - greenhouse_transition(model, state, minus, hour)
        ) / (2.0 * step)
    return a, b


def stabilizing_feedback(a, b, state_cost=1.0, control_cost=0.1):
    """Return the discrete LQR feedback in the convention ``u = v + K e``."""
    q = np.eye(a.shape[0]) * state_cost
    r = np.eye(b.shape[1]) * control_cost
    try:
        p = solve_discrete_are(a, b, q, r)
        return -np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    except Exception:
        return -0.25 * np.linalg.pinv(b)


def _slew_constraint(n_controls, slew, previous=None):
    rows, lower, upper = [], [], []
    for t in range(n_controls - 1):
        for j in range(4):
            row = np.zeros(n_controls * 4)
            row[t * 4 + j] = -1.0
            row[(t + 1) * 4 + j] = 1.0
            rows.append(row)
            lower.append(-slew[j])
            upper.append(slew[j])
    if previous is not None:
        for j in range(4):
            row = np.zeros(n_controls * 4)
            row[j] = 1.0
            rows.append(row)
            lower.append(previous[j] - slew[j])
            upper.append(previous[j] + slew[j])
    return LinearConstraint(np.asarray(rows), np.asarray(lower), np.asarray(upper))


def run_tube_rmpc(
    env,
    seed=42,
    prediction_horizon=6,
    max_iter=300,
    disturbance_radius=None,
    verbose=False,
):
    """Run receding-horizon tube RMPC with propagated robust constraints."""
    rng = np.random.default_rng(seed)
    cfg = env.config.base_config if hasattr(env.config, "base_config") else env.config
    actual_model = env.eval_env if hasattr(env, "eval_env") else env
    nominal_model = env.base_env if hasattr(env, "base_env") else env
    horizon = int(cfg.T_steps)
    full_low, full_high = env.bounds()
    low, high = np.asarray(full_low[:4]), np.asarray(full_high[:4])
    slew = np.array([cfg.max_dT, cfg.max_dL, cfg.max_dC, cfg.max_dH], dtype=float)
    midpoint = (low + high) / 2.0
    plan = rng.uniform(low, high, size=(horizon, 4))
    for t in range(1, horizon):
        plan[t] = np.clip(plan[t], plan[t - 1] - slew, plan[t - 1] + slew)

    nominal_state = _initial_state(nominal_model)
    actual_state = _initial_state(actual_model)
    disturbance = np.asarray(
        disturbance_radius or TubeRMPCConfig().disturbance_radius, dtype=float
    )
    total_evals = 0
    convergence = []
    gains, all_tubes = [], []
    t0 = perf_counter()

    for hour in range(horizon):
        local_horizon = min(prediction_horizon, horizon - hour)
        a, b = linearize_transition(nominal_model, nominal_state, midpoint, hour)
        gain = stabilizing_feedback(a, b)
        closed_loop = a + b @ gain
        tube = propagate_box(closed_loop, disturbance, local_horizon)
        input_radii = np.abs(gain) @ tube.T
        input_radii = input_radii.T
        maximum_radius = 0.45 * (high - low)
        input_radii = np.minimum(input_radii, maximum_radius)
        stage_low = np.vstack([low + radius for radius in input_radii])
        stage_high = np.vstack([high - radius for radius in input_radii])
        climate_low = np.array([cfg.T_lower, cfg.C_lower, cfg.H_lower], dtype=float)
        climate_high = np.array([cfg.T_upper, cfg.C_upper, cfg.H_upper], dtype=float)
        climate_radii = np.minimum(
            tube[:, :3], 0.45 * (climate_high - climate_low)
        )
        tightened_state_low = climate_low + climate_radii
        tightened_state_high = climate_high - climate_radii

        initial = plan[hour : hour + local_horizon].copy()
        initial = np.clip(initial, stage_low, stage_high)
        if hour:
            initial[0] = np.clip(initial[0], plan[hour - 1] - slew, plan[hour - 1] + slew)
        for k in range(1, local_horizon):
            initial[k] = np.clip(initial[k], initial[k - 1] - slew, initial[k - 1] + slew)

        def objective(flat):
            nonlocal total_evals
            candidate = plan.copy()
            candidate[hour : hour + local_horizon] = flat.reshape(local_horizon, 4)
            if hour + local_horizon < horizon:
                candidate[hour + local_horizon :] = midpoint
            value, _ = nominal_model.fitness(candidate)
            total_evals += 1
            return -float(value)

        def nominal_climate(flat):
            state = nominal_state.copy()
            predicted = []
            for offset, control in enumerate(flat.reshape(local_horizon, 4)):
                state = greenhouse_transition(nominal_model, state, control, hour + offset)
                predicted.append(state[:3])
            return np.asarray(predicted).reshape(-1)

        bounds = Bounds(stage_low.reshape(-1), stage_high.reshape(-1))
        constraint = _slew_constraint(
            local_horizon, slew, plan[hour - 1] if hour else None
        )
        state_constraint = NonlinearConstraint(
            nominal_climate,
            tightened_state_low.reshape(-1),
            tightened_state_high.reshape(-1),
        )
        result = minimize(
            objective,
            initial.reshape(-1),
            method="SLSQP",
            bounds=bounds,
            constraints=[constraint, state_constraint],
            options={"maxiter": max_iter, "ftol": 1e-6},
        )
        nominal_controls = (result.x if np.all(np.isfinite(result.x)) else initial.reshape(-1)).reshape(
            local_horizon, 4
        )
        error = actual_state - nominal_state
        applied = nominal_controls[0] + gain @ error
        applied = np.clip(applied, low, high)
        if hour:
            applied = np.clip(applied, plan[hour - 1] - slew, plan[hour - 1] + slew)
        plan[hour] = applied
        nominal_state = greenhouse_transition(
            nominal_model, nominal_state, nominal_controls[0], hour
        )
        actual_state = greenhouse_transition(actual_model, actual_state, applied, hour)
        gains.append(gain)
        all_tubes.append(tube)
        check, _ = env.fitness(plan)
        total_evals += 1
        convergence.append(float(check))
        if verbose:
            print(f"TubeRMPC hour={hour + 1}/{horizon}, objective={check:.6f}")

    final_fitness, details = env.fitness(plan)
    total_evals += 1
    perturbation_penalty = float(details.get("perturbation_total_penalty", 0.0))
    return TubeRMPCResult(
        best_fitness=float(final_fitness),
        best_yield=float(details.get("total_yield", 0.0)),
        best_energy=float(details.get("total_energy", 0.0)),
        best_penalty=float(details.get("total_penalty", 0.0)) + perturbation_penalty,
        feasible=bool(details.get("is_feasible", False)),
        n_evals=total_evals,
        elapsed=perf_counter() - t0,
        best_x=plan,
        convergence=convergence,
        feedback_gain=np.asarray(gains),
        tube_radii=np.asarray(all_tubes, dtype=object),
    )
