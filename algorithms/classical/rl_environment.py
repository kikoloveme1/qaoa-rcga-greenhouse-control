"""Sequential reinforcement-learning adapter for the greenhouse simulator."""

from __future__ import annotations

import numpy as np


class GreenhouseControlAdapter:
    """Expose one greenhouse day as 24 normalized continuous-control steps."""

    action_dim = 4
    state_dim = 14

    def __init__(self, env):
        self.env = env
        self.model = env.eval_env if hasattr(env, "eval_env") else env
        cfg = env.config.base_config if hasattr(env.config, "base_config") else env.config
        self.cfg = cfg
        low, high = env.bounds()
        self.low = np.asarray(low[:4], dtype=float)
        self.high = np.asarray(high[:4], dtype=float)
        self.slew = np.array([cfg.max_dT, cfg.max_dL, cfg.max_dC, cfg.max_dH], dtype=float)
        self.midpoint = (self.low + self.high) / 2.0
        self.interactions = 0
        self.reset()

    @property
    def horizon(self):
        return int(self.cfg.T_steps)

    def reset(self):
        self.hour = 0
        self.interactions = 0
        self.plan = np.tile(self.midpoint, (self.horizon, 1))
        initial_fitness, initial_details = self.env.fitness(self.plan)
        self.initial_fitness = float(initial_fitness)
        self.previous_fitness = self.initial_fitness
        self.details = initial_details
        return self._state(0, initial_details)

    def _state(self, hour, details):
        t = min(int(hour), self.horizon - 1)
        previous = self.midpoint if t == 0 else self.plan[t - 1]
        previous_norm = 2.0 * (previous - self.low) / (self.high - self.low) - 1.0

        def trajectory(name, default, scale):
            values = details.get(name)
            value = default if values is None else float(np.asarray(values)[t])
            return value / scale

        rh_out = getattr(self.model, "_RH_out", np.full(self.horizon, self.cfg.RH_out))
        state = np.array(
            [
                np.sin(2.0 * np.pi * t / 24.0),
                np.cos(2.0 * np.pi * t / 24.0),
                float(self.model._T_out[t]) / 40.0,
                float(rh_out[t]) / 100.0,
                float(self.model._solar[t]) / 1000.0,
                float(self.model._tou_prices[t]) / max(self.cfg.price_peak, 1e-9),
                trajectory("co2_realized", self.cfg.CO2_out, 1800.0),
                trajectory("rh_realized", self.cfg.RH_out, 100.0),
                trajectory("lai_hourly", self.cfg.LAI_initial, 5.0),
                trajectory("dry_matter_hourly", self.cfg.DM_initial, 500.0),
                *previous_norm,
            ],
            dtype=np.float32,
        )
        return state

    def _scale_action(self, action):
        normalized = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        control = self.low + (normalized + 1.0) * 0.5 * (self.high - self.low)
        if self.hour:
            control = np.clip(control, self.plan[self.hour - 1] - self.slew, self.plan[self.hour - 1] + self.slew)
        return np.clip(control, self.low, self.high)

    def step(self, action):
        if self.hour >= self.horizon:
            raise RuntimeError("episode is complete; call reset()")
        self.plan[self.hour] = self._scale_action(action)
        fitness, details = self.env.fitness(self.plan)
        fitness = float(fitness)
        reward = fitness - self.previous_fitness
        self.previous_fitness = fitness
        self.details = details
        self.interactions += 1
        self.hour += 1
        done = self.hour == self.horizon
        state = self._state(self.hour, details)
        info = dict(details)
        info.update(
            plan=self.plan.copy(),
            fitness=fitness,
            initial_fitness=self.initial_fitness,
            interactions=self.interactions,
        )
        return state, float(reward), done, info
