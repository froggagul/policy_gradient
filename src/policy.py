from typing import Dict, List, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gym import wrappers
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from src.render import AbstractRenderer


class MLPPolicy(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super(MLPPolicy, self).__init__()

        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.Tanh())

        modules.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*modules)

    def configure_optimizer(
        self, optimizer: torch.optim.Optimizer, learning_rate: float
    ):
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, x: torch.FloatTensor):
        return self.net(x)

    def get_policy(self, x: np.ndarray):
        x = torch.FloatTensor(x)
        logits = self(x)
        return Categorical(logits=logits)

    def get_action(self, x: np.ndarray):
        x = torch.FloatTensor(x)
        return self.get_policy(x).sample().item()


class VanillaPolicyGradient:
    def __init__(
        self, env: gym.Env, model: MLPPolicy, renderer: AbstractRenderer = None
    ):
        self.env = env
        self.model = model
        self.renderer = renderer
        self.env.reset()

    def render(self):
        if self.renderer:
            self.renderer.render(self.env)

    def reward_to_go(self, rewards: List[float], discount_factor=0.99) -> np.ndarray:
        r = np.full(len(rewards), discount_factor) ** np.arange(
            len(rewards)
        ) * np.array(rewards)
        r = r[::-1].cumsum()[::-1]
        discounted_rewards = r - r.mean()

        return discounted_rewards.copy()

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        observations = batch["observations"]
        actions = batch["actions"]
        weights = batch["weights"]

        loss = (-self.model.get_policy(observations).log_prob(actions) * weights).mean()

        return loss

    def train(
        self,
        epochs: int = 30,
        batch_size: int = 300,
    ):
        self.model.train()

        for epoch in range(epochs):
            loss, rewards, episode_lengths = self.train_one_epoch(batch_size)
            print(
                f"epoch: {epoch}\tloss: {loss}\treward: {np.mean(rewards)}\tep_len: {np.mean(episode_lengths)}\tnum_episodes: {len(rewards)}"
            )

    def collect_trajectories(self, batch_size: int):
        batch = {
            "observations": [],
            "actions": [],
            "weights": [],
            "lengths": [],
            "rewards": [],
        }

        while True:
            episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
            }
            obs, _ = self.env.reset()
            done = False

            while not done:
                episode["observations"].append(obs)

                action = self.model.get_action(obs)
                (obs, reward, done, *_) = self.env.step(action)
                episode["actions"].append(action)
                episode["rewards"].append(reward)

            # after done
            batch["rewards"].append(sum(episode["rewards"]))
            batch["lengths"].append(len(episode["rewards"]))

            weights = self.reward_to_go(episode["rewards"])
            batch["weights"].extend(weights)
            batch["observations"].extend(episode["observations"])
            batch["actions"].extend(episode["actions"])

            if len(batch["observations"]) >= batch_size:
                break

        batch["observations"] = np.array(batch["observations"])
        batch["observations"] = torch.FloatTensor(batch["observations"])
        batch["actions"] = torch.LongTensor(batch["actions"])
        batch["weights"] = torch.FloatTensor(batch["weights"])

        return batch

    def train_one_epoch(self, batch_size: int):
        batch = self.collect_trajectories(batch_size)

        self.model.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.model.optimizer.step()

        return loss, batch["rewards"], batch["lengths"]

    def test(self, n_episodes: int = 10):
        self.renderer.initialize(self.env)
        self.model.eval()
        rewards = []

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.render()
                action = self.model.get_action(obs)
                obs, reward, done, *_ = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)

        print("test finished")
        for episode, episode_reward in enumerate(rewards):
            print(f"\tEpisode: {episode} \t Reward: {episode_reward}")
