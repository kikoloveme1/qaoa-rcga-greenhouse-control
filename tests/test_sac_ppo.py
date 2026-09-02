import numpy as np
import torch

from algorithms.classical.sac_ppo import (
    SACPPOConfig,
    SquashedGaussianActor,
    ppo_clipped_loss,
    run_sac_ppo,
)
from environment.greenhouse_model import GreenhouseEnv


def test_article_budget_and_actor_bounds():
    assert SACPPOConfig().total_steps == 40_000
    torch.manual_seed(1)
    actor = SquashedGaussianActor(14, 4, hidden_dim=16)
    action, log_prob = actor.sample(torch.zeros(5, 14))

    assert action.shape == (5, 4)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
    assert log_prob.shape == (5, 1)


def test_ppo_objective_clips_large_policy_ratio():
    old_log_prob = torch.zeros(2, 1)
    new_log_prob = torch.log(torch.tensor([[2.0], [0.5]]))
    advantage = torch.tensor([[1.0], [-1.0]])

    loss = ppo_clipped_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2)

    assert torch.isclose(loss, torch.tensor(-0.2))


def test_sac_ppo_smoke_is_seeded_and_bounded():
    kwargs = dict(
        budget=32,
        hidden_dim=16,
        warmup_steps=4,
        batch_size=4,
        rollout_size=4,
        ppo_epochs=1,
    )
    first = run_sac_ppo(GreenhouseEnv(seed=8), seed=8, **kwargs)
    second = run_sac_ppo(GreenhouseEnv(seed=8), seed=8, **kwargs)
    low, high = GreenhouseEnv(seed=8).bounds()

    assert first.method == "SAC-PPO"
    assert first.n_evals == 32
    assert first.best_x.shape == (24, 4)
    assert np.all(first.best_x.reshape(-1) >= low)
    assert np.all(first.best_x.reshape(-1) <= high)
    assert np.allclose(first.best_x, second.best_x)
