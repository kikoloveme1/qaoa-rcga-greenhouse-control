import numpy as np

from algorithms.classical.es_policy_search import (
    ESPolicySearchResult,
    run_es_policy_search,
)
from environment.greenhouse_model import GreenhouseEnv


def test_es_policy_search_keeps_its_identity():
    assert ESPolicySearchResult().method == "ESPolicySearch"


def test_es_policy_search_returns_a_bounded_plan():
    env = GreenhouseEnv(seed=3)
    result = run_es_policy_search(env, seed=3, budget=3)
    low, high = env.bounds()

    assert result.best_x.shape == (24, 4)
    assert result.n_evals == 3
    assert np.all(result.best_x.reshape(-1) >= low)
    assert np.all(result.best_x.reshape(-1) <= high)
