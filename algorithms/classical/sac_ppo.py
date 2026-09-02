"""Hybrid Soft Actor-Critic and Proximal Policy Optimization controller."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from time import perf_counter
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .rl_environment import GreenhouseControlAdapter


@dataclass(frozen=True)
class SACPPOConfig:
    total_steps: int = 40_000
    hidden_dim: int = 128
    replay_capacity: int = 50_000
    warmup_steps: int = 512
    batch_size: int = 128
    rollout_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    clip_eps: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    reward_scale: float = 0.05
    max_grad_norm: float = 1.0


@dataclass
class SACPPOResult:
    method: str = "SAC-PPO"
    best_fitness: float = 0.0
    best_yield: float = 0.0
    best_energy: float = 0.0
    best_penalty: float = 0.0
    feasible: bool = False
    n_evals: int = 0
    elapsed: float = 0.0
    best_x: Optional[np.ndarray] = None
    convergence: list = field(default_factory=list)


def _mlp(input_dim, output_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = _mlp(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, state):
        hidden = self.backbone(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-5.0, 2.0)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(distribution, raw, action):
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(raw) - correction).sum(dim=-1, keepdim=True)

    def sample(self, state):
        distribution = self.distribution(state)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        return action, self._squashed_log_prob(distribution, raw, action)

    def deterministic(self, state):
        return torch.tanh(self.distribution(state).mean)

    def log_prob(self, state, action):
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self._squashed_log_prob(self.distribution(state), raw, action)


class SoftQCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = _mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        return self.network(torch.cat((state, action), dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.position = 0
        self.size = 0
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.integers(0, self.size, size=batch_size)
        return tuple(
            torch.as_tensor(values[indices], dtype=torch.float32)
            for values in (self.states, self.actions, self.rewards, self.next_states, self.dones)
        )


def ppo_clipped_loss(new_log_prob, old_log_prob, advantage, clip_eps):
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.minimum(ratio * advantage, clipped * advantage).mean()


class SACPPOAgent:
    def __init__(self, state_dim, action_dim, config, seed):
        torch.manual_seed(seed)
        self.config = config
        self.actor = SquashedGaussianActor(state_dim, action_dim, config.hidden_dim)
        self.critic1 = SoftQCritic(state_dim, action_dim, config.hidden_dim)
        self.critic2 = SoftQCritic(state_dim, action_dim, config.hidden_dim)
        self.target1 = SoftQCritic(state_dim, action_dim, config.hidden_dim)
        self.target2 = SoftQCritic(state_dim, action_dim, config.hidden_dim)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr)

    def act(self, state, deterministic=False):
        tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(tensor)
                log_prob = self.actor.log_prob(tensor, action)
            else:
                action, log_prob = self.actor.sample(tensor)
        return action.squeeze(0).numpy(), float(log_prob.item())

    def update_critics(self, batch):
        states, actions, rewards, next_states, dones = batch
        cfg = self.config
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q = torch.minimum(
                self.target1(next_states, next_actions), self.target2(next_states, next_actions)
            ) - cfg.alpha * next_log_prob
            bellman = rewards + cfg.gamma * (1.0 - dones) * target_q
        losses = []
        for critic, optimizer in (
            (self.critic1, self.critic1_optimizer),
            (self.critic2, self.critic2_optimizer),
        ):
            loss = nn.functional.mse_loss(critic(states, actions), bellman)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        self._update_targets()
        return losses

    def update_actor(self, rollout):
        states = torch.as_tensor(np.asarray([row[0] for row in rollout]), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray([row[1] for row in rollout]), dtype=torch.float32)
        old_log_prob = torch.as_tensor(np.asarray([[row[2]] for row in rollout]), dtype=torch.float32)
        with torch.no_grad():
            advantage = torch.minimum(self.critic1(states, actions), self.critic2(states, actions))
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        loss_value = 0.0
        for _ in range(self.config.ppo_epochs):
            new_log_prob = self.actor.log_prob(states, actions)
            policy_loss = ppo_clipped_loss(
                new_log_prob, old_log_prob, advantage, self.config.clip_eps
            )
            entropy_loss = self.config.alpha * new_log_prob.mean()
            loss = policy_loss + entropy_loss
            self.actor_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            loss_value = float(loss.item())
        return loss_value

    def _update_targets(self):
        tau = self.config.tau
        with torch.no_grad():
            for target, source in ((self.target1, self.critic1), (self.target2, self.critic2)):
                for target_parameter, parameter in zip(target.parameters(), source.parameters()):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)


def run_sac_ppo(
    env,
    seed=42,
    budget=40_000,
    hidden_dim=128,
    warmup_steps=512,
    batch_size=128,
    rollout_size=256,
    ppo_epochs=4,
    verbose=False,
):
    """Train the hybrid controller and return its deterministic daily plan."""
    if budget < env.n_steps:
        raise ValueError("budget must include at least one complete 24-hour evaluation rollout")
    torch.set_num_threads(max(1, int(os.environ.get("SAC_PPO_TORCH_THREADS", "1"))))
    rng = np.random.default_rng(seed)
    config = SACPPOConfig(
        total_steps=budget,
        hidden_dim=hidden_dim,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        rollout_size=rollout_size,
        ppo_epochs=ppo_epochs,
    )
    adapter = GreenhouseControlAdapter(env)
    agent = SACPPOAgent(adapter.state_dim, adapter.action_dim, config, seed)
    replay = ReplayBuffer(config.replay_capacity, adapter.state_dim, adapter.action_dim, rng)
    rollout = []
    convergence = []
    state = adapter.reset()
    training_steps = budget - adapter.horizon
    t0 = perf_counter()

    for step in range(training_steps):
        if step < config.warmup_steps:
            action = rng.uniform(-1.0, 1.0, adapter.action_dim).astype(np.float32)
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                old_log_prob = float(agent.actor.log_prob(state_tensor, action_tensor).item())
        else:
            action, old_log_prob = agent.act(state)
        next_state, reward, done, info = adapter.step(action)
        scaled_reward = reward * config.reward_scale
        replay.add(state, action, scaled_reward, next_state, done)
        rollout.append((state.copy(), action.copy(), old_log_prob))
        state = next_state

        if replay.size >= config.batch_size:
            agent.update_critics(replay.sample(config.batch_size))
        if len(rollout) >= config.rollout_size:
            agent.update_actor(rollout)
            rollout.clear()
        if done:
            convergence.append(float(info["fitness"]))
            state = adapter.reset()
        if verbose and (step + 1) % 1000 == 0:
            print(f"SAC-PPO interactions={step + 1}/{training_steps}")

    if rollout:
        agent.update_actor(rollout)

    state = adapter.reset()
    final_info = None
    for _ in range(adapter.horizon):
        action, _ = agent.act(state, deterministic=True)
        state, _, done, final_info = adapter.step(action)
    assert done and final_info is not None
    convergence.append(float(final_info["fitness"]))
    perturbation_penalty = float(final_info.get("perturbation_total_penalty", 0.0))
    elapsed = perf_counter() - t0
    return SACPPOResult(
        best_fitness=float(final_info["fitness"]),
        best_yield=float(final_info.get("total_yield", 0.0)),
        best_energy=float(final_info.get("total_energy", 0.0)),
        best_penalty=float(final_info.get("total_penalty", 0.0)) + perturbation_penalty,
        feasible=bool(final_info.get("is_feasible", False)),
        n_evals=budget,
        elapsed=elapsed,
        best_x=np.asarray(final_info["plan"], dtype=float),
        convergence=convergence,
    )
