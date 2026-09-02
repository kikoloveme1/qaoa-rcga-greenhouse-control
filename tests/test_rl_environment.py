import numpy as np
import pytest

from algorithms.classical.rl_environment import GreenhouseControlAdapter
from environment.greenhouse_model import GreenhouseEnv


def test_adapter_runs_exactly_one_day_and_tracks_dense_reward():
    adapter = GreenhouseControlAdapter(GreenhouseEnv(seed=4))
    state = adapter.reset()

    assert state.shape == (adapter.state_dim,)
    assert np.all(np.isfinite(state))

    total_reward = 0.0
    for hour in range(24):
        state, reward, done, info = adapter.step(np.zeros(4))
        total_reward += reward
        assert done is (hour == 23)

    assert info["plan"].shape == (24, 4)
    assert total_reward == pytest.approx(info["fitness"] - info["initial_fitness"])
    assert adapter.interactions == 24


def test_adapter_clips_actions_and_enforces_slew_limits():
    env = GreenhouseEnv(seed=5)
    adapter = GreenhouseControlAdapter(env)
    adapter.reset()
    adapter.step(np.array([-10.0, 10.0, 10.0, -10.0]))
    _, _, _, info = adapter.step(np.array([10.0, -10.0, -10.0, 10.0]))
    cfg = env.config

    assert np.all(
        np.abs(np.diff(info["plan"][:2], axis=0))[0]
        <= np.array([cfg.max_dT, cfg.max_dL, cfg.max_dC, cfg.max_dH]) + 1e-12
    )
