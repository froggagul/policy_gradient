import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from typing import List

from typing import Dict, List, Tuple, Union


class MLPPolicy(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super(MLPPolicy, self).__init__()

        modules = []
        for i in range(len(dims) - 2):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*modules)

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
    def __init__(self, env: gym.Env, model: MLPPolicy, render=False):
        self.env = env
        self.model = model
        self.render = render

    def configure_optimizer(
        self, optimizer: torch.optim.Optimizer, learning_rate: float
    ):
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def set_render(self, render: bool):
        self.render = render

    def render_if_available(self):
        if self.render:
            try:
                self.env.render()
            except Exception as e:
                print(e)

    def reward_to_go(self, rewards: List[float]) -> torch.Tensor:
        return torch.FloatTensor(np.cumsum(rewards[::-1])[::-1].copy())

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        observations = batch["observations"]
        actions = batch["actions"]
        weights = batch["weights"]

        self.optimizer.zero_grad()
        loss: torch.Tensor = (
            -self.model.get_policy(observations).log_prob(actions) * weights
        )

        return torch.mean(loss)

    def train(
        self,
        epochs: int = 30,
        batch_size: int = 5000,
    ):
        for epoch in range(epochs):
            loss, rewards, episode_lengths = self.train_one_epoch(batch_size)
            print(
                f"epoch: {epoch} \t loss: {loss} \t reward: {np.mean(rewards)} \t ep_len: {np.mean(episode_lengths)}"
            )

    def collect_trajectories(self, batch_size: int):
        batch = {
            "observations": [],
            "actions": [],
            "weights": [],
            "lengths": [],
            "rewards": [],
        }
        episode_rewards = []
        obs, _ = self.env.reset()
        done = False

        for _ in range(batch_size):
            self.render_if_available()
            action = self.model.get_action(obs)
            (obs, reward, done, *_) = self.env.step(action)

            batch["observations"].append(obs)
            batch["actions"].append(action)
            episode_rewards.append(reward)

            if done:
                batch["weights"].extend(self.reward_to_go(episode_rewards))
                obs, _ = self.env.reset()

                batch["lengths"].append(len(episode_rewards))
                batch["rewards"].append(sum(episode_rewards))

                episode_rewards = []
                done = False

        if not done and len(episode_rewards) > 0:
            batch["weights"].extend(self.reward_to_go(episode_rewards))
            batch["lengths"].append(len(episode_rewards))
            batch["rewards"].append(sum(episode_rewards))

        batch["observations"] = np.array(batch["observations"])

        batch["observations"] = torch.FloatTensor(batch["observations"])
        batch["actions"] = torch.LongTensor(batch["actions"])
        print(batch["weights"])
        batch["weights"] = torch.stack(batch["weights"], dim=0)

        return batch

    def train_one_epoch(self, batch_size: int):
        batch = self.collect_trajectories(batch_size)

        self.optimizer.zero_grad()
        loss = self.loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss, batch["rewards"], batch["lengths"]

    def test(self, n_episodes: int = 10):
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.render_if_available()
                action = self.model.get_action(obs)
                obs, reward, done, *_ = self.env.step(action)
                episode_reward += reward

            print(f"Episode: {episode} \t Reward: {episode_reward}")


if __name__ == "__main__":
    from tap import Tap

    class ArgumentParser(Tap):
        env_name: str = "CartPole-v1"
        render: bool = True
        lr: float = 1e-2

    parser = ArgumentParser()
    args = parser.parse_args()
    print(f"starting experiment with args: {args}")

    env = gym.make(args.env_name, render_mode="rgb_array")

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    model = MLPPolicy(
        [
            obs_dim,
            32,
            n_acts,
        ]
    )

    vpg = VanillaPolicyGradient(env, model, render=False)
    vpg.configure_optimizer(Adam, args.lr)
    vpg.train(epochs=30, batch_size=100)

    vpg.set_render(args.render)
    vpg.test()
