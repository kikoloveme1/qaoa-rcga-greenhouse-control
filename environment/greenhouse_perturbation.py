# -*- coding: utf-8 -*-
"""Weather and equipment perturbations for greenhouse controller evaluation.

Wraps greenhouse dynamics with modified outdoor profiles, equipment
availability, yield losses and energy surcharges."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy

from .greenhouse_model import GreenhouseEnv, GreenhouseConfig


class ScenarioType(Enum):
    """Categories of perturbation scenarios."""
    BASELINE = auto()
    HEAT_WAVE = auto()
    COLD_SNAP = auto()
    STORM = auto()
    HEATER_FAILURE = auto()
    CO2_FAILURE = auto()
    PEST_OUTBREAK = auto()
    COMPOUND = auto()
    ROLLING = auto()
    STOCHASTIC = auto()


@dataclass
class PerturbationScenario:
    """Defines a single perturbation event.

    Attributes
    ----------
    scenario_type : ScenarioType
    onset_hour : int  (0-23)
    duration_hours : int
    intensity : float  0.0-1.0
    outdoor_temp_delta : float  Applied to outdoor T (Celsius)
    solar_reduction : float  0.0-1.0 fraction blocked
    humidity_bias : float  Added to ambient RH (%)
    co2_multiplier : float
    heating_multiplier : float
    cooling_multiplier : float
    yield_penalty : float  0.0-1.0
    """
    scenario_type: ScenarioType = ScenarioType.BASELINE
    onset_hour: int = 6
    duration_hours: int = 6
    intensity: float = 0.5

    outdoor_temp_delta: float = 0.0
    solar_reduction: float = 0.0
    humidity_bias: float = 0.0
    co2_multiplier: float = 1.0
    heating_multiplier: float = 1.0
    cooling_multiplier: float = 1.0
    yield_penalty: float = 0.0

    @property
    def end_hour(self) -> int:
        return self.onset_hour + self.duration_hours

    def is_active(self, hour: int) -> bool:
        return self.onset_hour <= hour < self.end_hour

# Scenario factory functions


def make_heat_wave(onset=10, duration=8, intensity=1.0) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.HEAT_WAVE,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        outdoor_temp_delta=8.0, humidity_bias=0.0,
        cooling_multiplier=0.6, yield_penalty=0.0,
    )

def make_cold_snap(onset=2, duration=6, intensity=1.0) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.COLD_SNAP,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        outdoor_temp_delta=-5.0, solar_reduction=0.3,
        heating_multiplier=1.0, yield_penalty=0.0,
    )

def make_storm(onset=14, duration=4, intensity=0.9) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.STORM,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        outdoor_temp_delta=-2.0, solar_reduction=0.85,
        humidity_bias=15.0,
    )

def make_heater_failure(onset=4, duration=5, intensity=1.0) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.HEATER_FAILURE,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        heating_multiplier=0.0,
    )

def make_co2_failure(onset=8, duration=6, intensity=1.0) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.CO2_FAILURE,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        co2_multiplier=0.05, yield_penalty=0.0,
    )

def make_pest_outbreak(onset=6, duration=12, intensity=0.6) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.PEST_OUTBREAK,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        yield_penalty=0.3, humidity_bias=5.0,
    )

def make_compound_crisis(onset=10, duration=8, intensity=1.0) -> PerturbationScenario:
    return PerturbationScenario(
        scenario_type=ScenarioType.COMPOUND,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
        outdoor_temp_delta=8.0, co2_multiplier=0.05,
        cooling_multiplier=0.6, yield_penalty=0.30,
        humidity_bias=0.0,
    )


def make_rolling(onset=0, duration=24, intensity=1.0) -> PerturbationScenario:
    """Continuous rolling disturbance: temperature, solar, humidity and CO2 availability rotate every 4 hours."""
    return PerturbationScenario(
        scenario_type=ScenarioType.ROLLING,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
    )


def make_stochastic(onset=0, duration=24, intensity=1.0) -> PerturbationScenario:
    """Stochastic hourly weather fluctuations around the nominal Singapore profile."""
    return PerturbationScenario(
        scenario_type=ScenarioType.STOCHASTIC,
        onset_hour=onset, duration_hours=duration, intensity=intensity,
    )


SCENARIO_LIBRARY = {
    'baseline': lambda: PerturbationScenario(scenario_type=ScenarioType.BASELINE),
    'heat_wave': make_heat_wave,
    'cold_snap': make_cold_snap,
    'storm': make_storm,
    'heater_failure': make_heater_failure,
    'co2_failure': make_co2_failure,
    'pest_outbreak': make_pest_outbreak,
    'compound_crisis': make_compound_crisis,
    'rolling': make_rolling,
    'stochastic': make_stochastic,
}

# Perturbation Environment


@dataclass
class PerturbationConfig:
    base_config: GreenhouseConfig = field(default_factory=GreenhouseConfig)
    scenarios: List[PerturbationScenario] = field(default_factory=list)
    seed: int = 42

    # Delegate common GreenhouseConfig attributes for compatibility
    @property
    def T_steps(self): return self.base_config.T_steps
    @property
    def T_out_mean(self): return self.base_config.T_out_mean
    @property
    def RH_out(self): return self.base_config.RH_out


class PerturbationEnv:
    """Evaluate modified weather profiles and scenario economic adjustments."""

    def __init__(self, config=None, scenarios=None):
        self.config = config or PerturbationConfig()
        self.scenarios = scenarios or self.config.scenarios
        self.base_env = GreenhouseEnv(self.config.base_config, seed=self.config.seed)
        self._perturbation_rng = np.random.default_rng(self.config.seed + 777)
        self._co2_scale = np.ones(self.config.base_config.T_steps)
        self.eval_env = self._build_eval_env()

    def _build_eval_env(self) -> GreenhouseEnv:
        """Build the actual weather-perturbed evaluation environment.

        The base environment remains unperturbed and is used for QAOA/QUBO
        construction. This separate evaluation environment applies temperature,
        solar, humidity and CO2-availability perturbations directly to the
        weather profiles, rather than only using post-hoc penalty terms.
        """
        n = self.config.base_config.T_steps
        t_delta = np.zeros(n)
        solar_red = np.zeros(n)
        solar_delta = np.zeros(n)
        rh_bias = np.zeros(n)
        co2_scale = np.ones(n)

        for sc in self.scenarios:
            if sc.scenario_type == ScenarioType.STOCHASTIC:
                rng = np.random.default_rng(self.config.seed + 555)
                t_delta += np.clip(rng.normal(0.0, 2.0, n), -5.0, 5.0)
                solar_delta += np.clip(rng.normal(0.0, 50.0, n), -150.0, 150.0)
                rh_bias += np.clip(rng.normal(0.0, 5.0, n), -12.5, 12.5)
                continue

            if sc.scenario_type == ScenarioType.ROLLING:
                for t in range(n):
                    phase = (t // 4) % 4
                    if phase == 0:
                        t_delta[t] += 5.0 * sc.intensity
                    elif phase == 1:
                        solar_red[t] = max(solar_red[t], 0.30 * sc.intensity)
                    elif phase == 2:
                        rh_bias[t] += 15.0 * sc.intensity
                    else:
                        co2_scale[t] *= 0.80
                continue

            for t in range(n):
                if sc.is_active(t):
                    t_delta[t] += sc.outdoor_temp_delta * sc.intensity
                    solar_red[t] = max(solar_red[t], sc.solar_reduction * sc.intensity)
                    rh_bias[t] += sc.humidity_bias * sc.intensity
                    if sc.co2_multiplier != 1.0:
                        co2_scale[t] *= sc.co2_multiplier

        # Clone the weather profiles and apply the real perturbations.
        cfg = deepcopy(self.config.base_config)
        eval_env = GreenhouseEnv(cfg, seed=self.config.seed + 999)
        eval_env._T_out = self.base_env._T_out + t_delta
        eval_env._solar = np.maximum(0.0, self.base_env._solar * (1.0 - solar_red) + solar_delta)
        eval_env._RH_out = np.full(n, cfg.RH_out) + rh_bias
        eval_env._co2_injection_scale = co2_scale
        self._co2_scale = co2_scale
        return eval_env

    def fitness(self, plan: np.ndarray):
        """Evaluate plan under the weather-perturbed environment.

        CO2 injection is scaled hour-by-hour when the scenario includes CO2
        availability loss. Yield penalties and equipment energy surcharges are
        applied only when the scenario explicitly defines them.
        """
        plan = np.asarray(plan, dtype=float)
        if plan.size == self.config.base_config.T_steps * 4:
            plan = plan.reshape(self.config.base_config.T_steps, 4)

        base_f, base_info = self.eval_env.fitness(plan)

        n = self.config.base_config.T_steps
        yield_hourly = base_info.get('yield_hourly', None)
        energy_hourly = base_info.get('energy_hourly', None)
        total_yield = float(base_info.get('total_yield', 0.0))

        yield_loss = 0.0
        energy_surcharge = 0.0
        surcharge_hourly = np.zeros(n)
        active_hours = 0

        for sc in self.scenarios:
            if sc.scenario_type in (ScenarioType.STOCHASTIC, ScenarioType.ROLLING):
                active_hours = n
                continue

            for t in range(n):
                if sc.is_active(t):
                    active_hours += 1
                    intensity = sc.intensity

                    if sc.yield_penalty != 0.0:
                        if yield_hourly is not None and t < len(yield_hourly):
                            yield_loss += float(yield_hourly[t]) * sc.yield_penalty * intensity
                        else:
                            yield_loss += (total_yield / max(n, 1)) * sc.yield_penalty * intensity

                    if energy_hourly is not None and t < len(energy_hourly):
                        et = float(energy_hourly[t])
                        if sc.heating_multiplier < 1.0:
                            energy_surcharge += et * (1.0 - sc.heating_multiplier) * intensity
                            surcharge_hourly[t] += et * (1.0 - sc.heating_multiplier) * intensity
                        if sc.cooling_multiplier < 1.0:
                            energy_surcharge += et * (1.0 - sc.cooling_multiplier) * intensity
                            surcharge_hourly[t] += et * (1.0 - sc.cooling_multiplier) * intensity

        # Yield loss is in simulator output units, not SGD. Apply the same
        # conversion and crop price as the base model before changing Profit.
        cfg = self.base_env.config
        if any(sc.scenario_type == ScenarioType.COMPOUND for sc in self.scenarios):
            # Manuscript explicitly describes a 30% DAILY crop loss.
            yield_loss = total_yield * max(sc.yield_penalty * sc.intensity for sc in self.scenarios)
        revenue_loss = yield_loss * 0.04 * cfg.crop_market_price
        profit = float(base_info['profit_sgd']) - revenue_loss - energy_surcharge
        pert_fitness = profit - float(base_info['total_penalty'])

        info = dict(base_info)
        info['perturbation_yield_loss'] = yield_loss
        info['perturbation_energy_surcharge'] = energy_surcharge
        info['perturbation_total_penalty'] = revenue_loss + energy_surcharge
        info['perturbation_revenue_loss_sgd'] = revenue_loss
        info['total_yield'] = total_yield - yield_loss
        info['yield_kg_m2'] = info['total_yield'] * 0.04
        info['revenue_sgd'] = float(base_info['revenue_sgd']) - revenue_loss
        info['total_energy'] = float(base_info['total_energy']) + energy_surcharge
        info['profit_sgd'] = profit
        info['fitness'] = pert_fitness
        info['yield_hourly'] = np.asarray(base_info['yield_hourly']) * (1.0-yield_loss/max(total_yield,1e-12))
        info['energy_hourly'] = np.asarray(base_info['energy_hourly']) + surcharge_hourly
        info['active_hours'] = active_hours
        info['scenario_types'] = [sc.scenario_type.name for sc in self.scenarios]

        return pert_fitness, info

    def bounds(self):
        return self.base_env.bounds()

    @property
    def n_steps(self):
        return self.base_env.n_steps

    @property
    def n_vars(self):
        return self.base_env.n_vars

    @property
    def config_attr(self):
        return self.base_env.config

    def set_penalty_weights(self, lambda_rate: float, lambda_bound: float):
        """Apply penalty weights to both the QUBO/base and evaluation models.

        ``eval_env`` is a deep copy built at construction time, so callers that
        temporarily raise RCGA penalty weights must update it explicitly.
        """
        self.base_env.config.lambda_rate = lambda_rate
        self.base_env.config.lambda_bound = lambda_bound
        self.eval_env.config.lambda_rate = lambda_rate
        self.eval_env.config.lambda_bound = lambda_bound

    def perturbation_summary(self) -> str:
        lines = [f"PerturbationEnv with {len(self.scenarios)} scenario(s):"]
        for i, sc in enumerate(self.scenarios):
            onset = f"{sc.onset_hour:02d}:00"
            end = f"{sc.end_hour:02d}:00"
            lines.append(
                f"  [{i}] {sc.scenario_type.name}: {onset}-{end} "
                f"(intensity={sc.intensity:.1f}, dur={sc.duration_hours}h)"
            )
        return '\n'.join(lines)

    def random_solution(self):
        """Generate a random feasible-ish solution."""
        return self.base_env.random_solution()

    def summary(self, plan):
        f, info = self.fitness(plan)
        return (
            f"{self.perturbation_summary()}\n"
            f"Fitness: {f:.2f} | Yield: {info.get('yield', '?'):.3f} | "
            f"Energy: {info.get('energy', '?'):.1f} | "
            f"Penalty: {info.get('penalty', 0.0):.2f}"
        )

# Benchmark suite


def get_benchmark_suite(seed: int = 42) -> Dict[str, PerturbationEnv]:
    """Standard benchmark suite with a reproducible weather/disturbance seed."""

    def make_environment(scenarios):
        config = PerturbationConfig(
            base_config=GreenhouseConfig(),
            scenarios=list(scenarios),
            seed=int(seed),
        )
        return PerturbationEnv(config=config, scenarios=list(scenarios))

    return {
        'baseline': make_environment([SCENARIO_LIBRARY['baseline']()]),
        'heat_wave': make_environment(
            [make_heat_wave(onset=10, duration=8, intensity=1.0)]
        ),
        'cold_snap': make_environment(
            [make_cold_snap(onset=2, duration=6, intensity=1.0)]
        ),
        'co2_failure': make_environment(
            [make_co2_failure(onset=8, duration=6, intensity=1.0)]
        ),
        'compound_crisis': make_environment(
            [make_compound_crisis(onset=10, duration=8, intensity=1.0)]
        ),
        'rolling': make_environment(
            [make_rolling(onset=0, duration=24, intensity=1.0)]
        ),
        'stochastic': make_environment(
            [make_stochastic(onset=0, duration=24, intensity=1.0)]
        ),
    }


if __name__ == '__main__':
    print("=== PerturbationEnv Quick Test ===\n")
    suite = get_benchmark_suite()
    for name, env in suite.items():
        print(f"--- {name} ---")
        print(env.perturbation_summary())
        low, high = env.bounds()
        plan = np.random.default_rng(42).uniform(low, high)
        f, info = env.fitness(plan)
        print(f"  Fitness: {f:.2f} | Yield: {info.get('yield', '?'):.3f} | "
              f"Energy: {info.get('energy', '?'):.1f} | "
              f"PertPenalty: {info.get('perturbation_penalty', 0):.2f}")
        print()
    print("All perturbation scenarios loaded.")
