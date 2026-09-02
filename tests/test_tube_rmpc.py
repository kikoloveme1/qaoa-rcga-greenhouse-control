import numpy as np

from algorithms.classical.tube_rmpc import (
    TubeRMPCConfig,
    propagate_box,
    run_tube_rmpc,
    tighten_box,
)
from environment.greenhouse_model import GreenhouseEnv


def test_box_tube_propagation_and_tightening():
    radius = propagate_box(np.eye(2) * 0.5, np.array([0.2, 0.1]), 3)
    assert np.allclose(radius[-1], [0.35, 0.175])

    low, high = tighten_box(
        np.array([-1.0, -2.0]), np.array([1.0, 2.0]), radius[-1]
    )
    assert np.allclose(low, [-0.65, -1.825])
    assert np.allclose(high, [0.65, 1.825])


def test_tube_rmpc_article_defaults_and_smoke_plan():
    assert TubeRMPCConfig().max_iter == 300
    assert TubeRMPCConfig().prediction_horizon == 6
    env = GreenhouseEnv(seed=7)
    first = run_tube_rmpc(env, seed=7, prediction_horizon=2, max_iter=3)
    second = run_tube_rmpc(GreenhouseEnv(seed=7), seed=7, prediction_horizon=2, max_iter=3)
    low, high = env.bounds()

    assert first.method == "TubeRMPC"
    assert np.allclose(first.best_x, second.best_x)
    assert np.all(first.best_x.reshape(-1) >= low)
    assert np.all(first.best_x.reshape(-1) <= high)
    assert np.all(
        np.abs(np.diff(first.best_x, axis=0))
        <= np.array([env.config.max_dT, env.config.max_dL, env.config.max_dC, env.config.max_dH])
        + 1e-8
    )
