import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import flappy_bird_gymnasium  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from envs.environment_handler import OurEnvironmentHandler
from gymnasium.wrappers import TimeLimit
from torch.distributions import Categorical
from torch.utils.tensorboard.writer import SummaryWriter

torch.manual_seed(0)
np.random.seed(0)

EPOCHS: int = 1000
NUM_ENVS = 10
LR: float = 0.0005
ROLLOUT_STEPS: int = 1024
GAMMA: float = 0.99
UPDATE_EPOCHS: int = 128
CLIP_COEF: float = 0.2
VF_COEF: float = 0.5
ENT_COEF: float = 0.01
GRAD_NORM: float = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS_PER_EPISODE = 1000
MODIFICATION = True
MIN_MODIFICATION = 0
MAX_MODIFICATION = 0.08
DECAY_MODIFICATION = 1
STANDARD_LEAKY = False


class ActorCritic(nn.Module):
    def __init__(self, state_space: int, action_space: int) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def act(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            action_probs: torch.Tensor = self.actor(state)
            state_value: torch.Tensor = self.critic(state).squeeze(1)
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, state_value

    def act_deterministically(self, state: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            action_probs: torch.Tensor = self.actor(state)
        return action_probs.argmax(dim=-1)

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_probs = self.actor(state)
        state_value = self.critic(state).squeeze(1)
        dist = Categorical(probs=action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, state_value, entropy


class RolloutBuffer:
    """
    Memory buffer, will be used to store the agent's experiences from the interaction stage.
    We will be learning from them in the training stage.
    It's tuples (s, a, r, log(pi(a|s)), V(s)), and an indicator for whether s is terminal
    """

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.state_values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def clear(self) -> None:
        self.__init__()


class Agent:
    """The agent that we are going to train with PPO"""

    def __init__(self, state_space: int, action_space: int, params: dict) -> None:
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_space, action_space).to(DEVICE)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(), lr=LR)
        for key, value in params.items():
            setattr(self, key, value)

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        """
        Interact with the environment. Store the experience
        Input:
            state: given from env.step()
        Return:
            action: to be passed into env.step() again
        """

        state: torch.Tensor = torch.from_numpy(state).float().unsqueeze(0)
        action, log_prob, state_value = self.policy.act(state=state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.log_probs.append(log_prob)
        self.buffer.state_values.append(state_value)

        return action.numpy()[0]

    def get_action_deterministic(self, state: np.ndarray) -> int:
        """Choose an action without storing in the rollout buffer."""
        state_t = torch.from_numpy(state).float().unsqueeze(0)
        with torch.inference_mode():
            action_probs = self.policy.actor(state_t)
        action = action_probs.argmax(dim=-1)
        return action.item()

    def choose_action_batch(self, states: np.ndarray):
        """
        For a batch of states of shape [N, state_dim],
        return a batch of actions of shape [N], etc.
        """
        states_t = torch.from_numpy(states).float().to(DEVICE)  # shape [N, state_dim]
        actions, log_probs, state_values = self.policy.act(states_t)

        self.buffer.states.append(states_t.cpu())  # Move back to CPU for storage
        self.buffer.actions.append(actions.cpu())
        self.buffer.log_probs.append(log_probs.cpu())
        self.buffer.state_values.append(state_values.cpu())

        return actions.detach().cpu().numpy()

    def store_outcome_batch(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """
        Store reward and done flags for a batch of environments (shape [N]).
        """
        self.buffer.rewards.append(rewards)  # shape [N]
        self.buffer.dones.append(dones)  # shape [N]

    def learn(self, global_step) -> None:
        """
        Flatten the entire rollout buffer and compute advantages in a vectorized way.
        In this example, we are collecting exactly 1 "full episode" in each of the N parallel envs,
        so the length of each environment's trajectory can differ. We'll handle that carefully.
        """
        # Flatten states, actions, log_probs, etc.
        # states is a list of length T (timesteps), each shape [N, state_dim]
        # We want [T*N, state_dim]
        old_states = torch.cat(self.buffer.states, dim=0).to(DEVICE)  # Move to DEVICE
        old_actions = torch.cat(self.buffer.actions, dim=0).to(DEVICE)
        old_log_probs = torch.cat(self.buffer.log_probs, dim=0).to(DEVICE)
        old_state_values = torch.cat(self.buffer.state_values, dim=0).to(DEVICE)

        rewards = torch.tensor(
            np.concatenate(self.buffer.rewards, axis=0),
            dtype=torch.float32,
            device=DEVICE,
        )
        dones = torch.tensor(
            np.concatenate(self.buffer.dones, axis=0),
            dtype=torch.float32,
            device=DEVICE,
        )

        # Because each environment can terminate at different times, we need to compute
        # the returns/advantages per environment. A simple way:
        #   - Unroll each env’s trajectory from the flattened buffer
        #   - If done[i] is True, we reset the discount the next step
        # Alternatively, if you're sure each env is exactly "one episode" and ends simultaneously,
        # you can do a simpler approach. But let's assume episodes can end at different steps.

        # We'll reconstruct the shape [T, N] for rewards and dones to do per-env GAE or discounted sum.
        T = len(self.buffer.rewards)  # timesteps collected
        N = rewards.size(0) // T
        rewards = rewards.reshape(T, N)
        dones = dones.reshape(T, N)
        state_values = old_state_values.view(T, N)  # same shape

        returns = torch.zeros_like(rewards, device=DEVICE)
        discounted_return = torch.zeros(N, device=DEVICE)

        for t in reversed(range(T)):
            discounted_return = rewards[t] + 0.99 * discounted_return * (1 - dones[t])
            returns[t] = discounted_return

        returns = returns.flatten()
        returns_torch = returns.clone().detach()

        state_values_flat = state_values.flatten()
        advantages = returns_torch - state_values_flat
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(UPDATE_EPOCHS):
            log_probs, state_values_new, entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = (log_probs - old_log_probs).exp()

            if self.modification:
                if self.leaky:
                    decay_coeff = 0.01
                else:
                    decay_coeff = max(
                        self.min_modification,
                        self.max_modification
                        - (self.max_modification - self.min_modification)
                        * (
                            self.decay_modification
                            * global_step
                            / (EPOCHS * NUM_ENVS * ROLLOUT_STEPS)
                        ),
                    )

                lower_bound = decay_coeff * ratios + (1 - decay_coeff) * (1 - CLIP_COEF)
                upper_bound = decay_coeff * ratios + (1 - decay_coeff) * (1 + CLIP_COEF)
                surr1 = advantages * ratios
                surr2 = advantages * torch.clamp(ratios, lower_bound, upper_bound)

            else:
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - CLIP_COEF, 1 + CLIP_COEF) * advantages

            critic_loss = F.mse_loss(state_values_new, returns_torch)

            loss = -torch.min(surr1, surr2) + VF_COEF * critic_loss - ENT_COEF * entropy
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), GRAD_NORM)
            self.optimizer.step()

        self.buffer.clear()


def evaluate_agent(agent: Agent, env: gym.Env) -> Tuple[float, float]:
    """
    Evaluate the agent by running it for a few episodes with a step limit.
    """
    returns = []
    lengths = []
    for i in range(1):
        if agent.env_name == "FlappyBird-v0":
            env = TimeLimit(env, max_episode_steps=MAX_STEPS_PER_EPISODE)
        state, _ = env.reset()
        done = False
        current_return = 0
        episode_length = 0
        while not done:
            episode_length += 1
            action = agent.get_action_deterministic(state)

            state, reward, terminated, truncated, _ = env.step(action)
            current_return += reward
            done = terminated or truncated
        returns.append(current_return)
        lengths.append(episode_length)
    return np.mean(returns), np.mean(lengths)


class Trainer:
    def __init__(self, env_name: str = "FlappyBird-v0", **hyperparams):
        """
        Initialize trainer with flexible hyperparameters.
        """

        self.defaults = {
            "gamma": 0.95,
            "learning_rate": 0.0005,
            "modification": False,
            "env_name": env_name,
            "leaky": False,
            "min_modification": 0,
            "max_modification": 0.08,
            "decay_modification": 1,
        }

        self.params = {**self.defaults, **hyperparams}
        self.agent = None
        for key, value in self.params.items():
            setattr(self, key, value)

    def train(
        self, epochs: int = EPOCHS, rollout_steps: int = ROLLOUT_STEPS
    ) -> List[float]:
        env_handler = OurEnvironmentHandler(
            env_id=self.env_name,
            num_envs=NUM_ENVS,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        )
        self.envs = env_handler.envs

        self.agent = Agent(env_handler.state_dim, env_handler.action_dim, self.params)

        global_step = 0

        writer_tb = SummaryWriter(
            log_dir=f"runs/VEC_PPO_{self.env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        episode_lengths = []
        avg_returns = []

        for update in range(epochs):
            states, _ = self.envs.reset()

            for i in range(rollout_steps):
                global_step += NUM_ENVS
                actions = self.agent.choose_action_batch(states)
                next_states, rewards, terminated, truncated, infos = self.envs.step(
                    actions
                )
                done_mask = (terminated | truncated).astype(float)
                self.agent.store_outcome_batch(rewards, done_mask)
                states = next_states

            self.agent.learn(global_step)

            env_eval = env_handler._make_env(seed=i)()
            avg_return, avg_length = evaluate_agent(self.agent, env_eval)
            env_eval.close()

            episode_lengths.append(avg_length)
            avg_returns.append(avg_return)
            writer_tb.add_scalar("rollout/ep_len_mean", avg_length, update)
            writer_tb.add_scalar("rollout/ep_rew_mean", avg_return, update)

            print(
                f"Update {update}: avg_return = {avg_return}, avg_length = {avg_length}"
            )

        return episode_lengths, avg_returns


class HyperparameterTuner:
    def __init__(self, env_name: str = "FlappyBird-v0", runs: int = 5):
        self.env_name = env_name
        self.runs = runs
        self.results = {}

    def run_experiment(self, **hyperparams) -> List[List[float]]:
        """Run multiple training sessions for the given hyperparameters and save results to CSV."""
        lengths = []
        returns = []

        param_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])

        for run in range(self.runs):
            print(f"Run {run + 1} with parameters: {param_str}")
            trainer = Trainer(**hyperparams, env_name=self.env_name)
            episode_lengths, episode_rewards = trainer.train()
            lengths.append(episode_lengths)
            returns.append(episode_rewards)

        filename_parts = []
        for key, value in hyperparams.items():
            if isinstance(value, bool):
                if value:
                    filename_parts.append(f"{key}")
            else:
                filename_parts.append(f"{key}_{str(value).replace('.', '_')}")

        filename = "results_" + "_".join(filename_parts) + ".csv"

        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Run", "Update", "Episode_Length", "Episode_Return"])
            for run_idx, (lengths_run, returns_run) in enumerate(zip(lengths, returns)):
                for update_idx, (length, reward) in enumerate(
                    zip(lengths_run, returns_run)
                ):
                    writer.writerow([run_idx + 1, update_idx + 1, length, reward])

        print(f"Results saved to {filename}")
        return lengths, returns

    def aggregate_results(
        self, lengths: List[List[float]], returns: List[List[float]]
    ) -> Dict[str, np.ndarray]:
        """Compute mean and standard deviation of the results."""
        lengths = np.array(lengths)  # Shape [runs, updates]
        mean_lengths = np.mean(lengths, axis=0)
        std_lengths = np.std(lengths, axis=0)

        returns = np.array(returns)  # Shape [runs, updates]
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)

        return {
            "mean_lengths": mean_lengths,
            "std_lengths": std_lengths,
            "mean_returns": mean_returns,
            "std_returns": std_returns,
        }

    def tune(self, hyperparams: List[Dict[str, float]]) -> None:
        """Tune the hyperparameters and store the results."""
        for params in hyperparams:
            lengths, returns = self.run_experiment(**params)
            stats = self.aggregate_results(lengths, returns)

            label_parts = []
            for key, value in params.items():
                if isinstance(value, bool):
                    if value:
                        label_parts.append(f"{key}")
                else:
                    label_parts.append(f"{key}={value}")
            label = ", ".join(label_parts)

            self.results[label] = stats

    @staticmethod
    def merge_separate_numbers(parts: List[str]) -> List[str]:
        merged = []
        skip_next = False

        def is_all_digits(s: str) -> bool:
            return all(ch.isdigit() for ch in s)

        for i in range(len(parts)):
            if skip_next:
                skip_next = False
                continue
            if (
                i + 1 < len(parts)
                and is_all_digits(parts[i])
                and is_all_digits(parts[i + 1])
            ):
                merged.append(parts[i] + "_" + parts[i + 1])
                skip_next = True
            else:
                merged.append(parts[i])
        return merged

    def read_results_from_csv(
        self, csv_files: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Reads the results from the given CSV files and aggregates them,
        creating nicer labels for well-known hyperparameters.
        Any unknown tokens will simply be appended as-is (e.g. 'modification' -> 'modification').
        Returns a dict mapping from the parsed label to another dict containing mean/std for lengths and returns.
        """
        KEY_MAPPING = {
            "gamma": "γ",
            "lr": "LR",
            "maxmod": "α(max)",
            "minmod": "α(min)",
            "decaymod": "k",
            "leaky": "leaky",
            "mod": "mod",
        }

        def combine_numeric_parts(*parts: str) -> str:
            joined = ".".join(parts)
            as_float = float(joined)
            return f"{as_float:g}"

        def is_all_digits(s: str) -> bool:
            return all(ch.isdigit() for ch in s)

        results = {}
        for csv_file in csv_files:
            data = pd.read_csv(csv_file)
            base_name = os.path.splitext(os.path.basename(csv_file))[0]

            parts = base_name.split("_")[1:]

            if not parts or (len(parts) == 1 and not parts[0]):
                label = "standard ppo"
            else:
                parts = self.merge_separate_numbers(parts)

                label_parts = []
                i = 0
                while i < len(parts):
                    token = parts[i]
                    if token in KEY_MAPPING:
                        display_key = KEY_MAPPING[token]

                        if i + 1 < len(parts):
                            next_token = parts[i + 1]
                            if "_" in next_token:
                                subparts = next_token.split("_")
                                if all(is_all_digits(sp) for sp in subparts):
                                    numeric_str = combine_numeric_parts(*subparts)
                                    label_parts.append(f"{display_key}={numeric_str}")
                                    i += 2
                                    continue
                            if is_all_digits(next_token):
                                label_parts.append(f"{display_key}={next_token}")
                                i += 2
                                continue
                        label_parts.append(display_key)
                        i += 1
                    else:
                        if "_" in token:
                            subparts = token.split("_")
                            if all(is_all_digits(sp) for sp in subparts):
                                numeric_str = combine_numeric_parts(*subparts)
                                label_parts.append(numeric_str)
                                i += 1
                                continue
                        label_parts.append(token)
                        i += 1

                label = ", ".join(label_parts)

            grouped = data.groupby("Update")
            mean_lengths = grouped["Episode_Length"].mean().values
            std_lengths = grouped["Episode_Length"].std().values
            mean_returns = grouped["Episode_Return"].mean().values
            std_returns = grouped["Episode_Return"].std().values

            results[label] = {
                "mean_lengths": mean_lengths,
                "std_lengths": std_lengths,
                "mean_returns": mean_returns,
                "std_returns": std_returns,
            }

        return results

    @staticmethod
    def smooth_data(data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Smooth the data using a moving average.
        Args:
            data: The input data (1D array).
            window_size: The size of the moving average window.
        Returns:
            Smoothed data as a 1D array.
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    def plot_results(
        self,
        save_name: str,
        title_lengths: str = "Mean Episode Length with SEM",
        title_returns: str = "Mean Episode Returns with SEM",
        save_folder: str = "plots",
        smoothing_window: int = 25,
    ) -> None:
        """Plot the mean and standard deviation for episode lengths and returns, and save the plots with smoothing."""

        os.makedirs(save_folder, exist_ok=True)

        plt.figure(figsize=(10, 6))
        for label, data in self.results.items():
            updates = len(data["mean_lengths"])
            timesteps = np.arange(updates) * ROLLOUT_STEPS * NUM_ENVS

            smoothed_mean_lengths = self.smooth_data(
                data["mean_lengths"], smoothing_window
            )
            smoothed_std_lengths = self.smooth_data(
                data["std_lengths"], smoothing_window
            )

            pad = (smoothing_window - 1) // 2
            smoothed_timesteps = timesteps[pad : -(pad + 1) if pad > 0 else None]

            min_len = min(len(smoothed_timesteps), len(smoothed_mean_lengths))
            smoothed_timesteps = smoothed_timesteps[:min_len]
            smoothed_mean_lengths = smoothed_mean_lengths[:min_len]
            smoothed_std_lengths = smoothed_std_lengths[:min_len]

            plt.plot(smoothed_timesteps, smoothed_mean_lengths, label=label)
            plt.fill_between(
                smoothed_timesteps,
                smoothed_mean_lengths - smoothed_std_lengths,
                smoothed_mean_lengths + smoothed_std_lengths,
                alpha=0.2,
            )

        plt.title(title_lengths)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Length")
        plt.legend(title="Hyperparameters")
        plt.grid()
        plot_path_lengths = os.path.join(save_folder, save_name + "_lengths.png")
        plt.savefig(plot_path_lengths)
        plt.close()

        plt.figure(figsize=(10, 6))
        for label, data in self.results.items():
            updates = len(data["mean_returns"])
            timesteps = np.arange(updates) * ROLLOUT_STEPS * NUM_ENVS

            smoothed_mean_returns = self.smooth_data(
                data["mean_returns"], smoothing_window
            )
            smoothed_std_returns = self.smooth_data(
                data["std_returns"], smoothing_window
            )

            pad = (smoothing_window - 1) // 2
            smoothed_timesteps = timesteps[pad : -(pad + 1) if pad > 0 else None]

            min_len = min(len(smoothed_timesteps), len(smoothed_mean_returns))
            smoothed_timesteps = smoothed_timesteps[:min_len]
            smoothed_mean_returns = smoothed_mean_returns[:min_len]
            smoothed_std_returns = smoothed_std_returns[:min_len]

            plt.plot(smoothed_timesteps, smoothed_mean_returns, label=label)
            plt.fill_between(
                smoothed_timesteps,
                smoothed_mean_returns - smoothed_std_returns,
                smoothed_mean_returns + smoothed_std_returns,
                alpha=0.2,
            )

        plt.title(title_returns)
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Returns")
        plt.legend(title="Hyperparameters")
        plt.grid()
        plot_path_returns = os.path.join(save_folder, save_name + "_returns.png")
        plt.savefig(plot_path_returns)
        plt.close()
        print(f"Plots saved to '{save_folder}'")


if __name__ == "__main__":
    hyperparams = [
        # {"gamma": 0.95, "learning_rate": 0.0005, "leaky": True},
        {
            "gamma": 0.95,
            "learning_rate": 0.0005,
            "modification": True,
            "max_modification": 0.05,
            "decay_modification": 1,
        },
    ]

    tuner = HyperparameterTuner(env_name="TradingEnv", runs=3)
    tuner.tune(hyperparams)
    # tuner.results = tuner.read_results_from_csv(
    #     [
    #         "results_gamma_0_95_learning_rate_0_005.csv",
    #         "results_gamma_0_95_learning_rate_0_0005.csv",
    #         "results_gamma_0_99_learning_rate_0_005.csv",
    #         "results_gamma_0_99_learning_rate_0_0005.csv",
    #     ]
    # )
    # tuner.plot_results("TradingEnv_hyperparams_tuning")
