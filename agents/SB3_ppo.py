from dataclasses import dataclass

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.environment_handler import SB3EnvironmentHandler


class SaveTrainingMetricsCallback(BaseCallback):
    """
    Custom callback for saving training metrics to a CSV file and generating plots.
    """

    def __init__(self, log_path: str = None, verbose: int = 1):
        super(SaveTrainingMetricsCallback, self).__init__(verbose)
        self.log_path = log_path
        self.metrics = []

    def _on_step(self) -> bool:
        """
        Called at every step during training.
        """
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            ep_lengths = [ep_info["l"] for ep_info in self.model.ep_info_buffer]
            ep_rew_mean = np.mean(ep_rewards)
            ep_len_mean = np.mean(ep_lengths)

            self.metrics.append(
                {
                    "timesteps": self.num_timesteps,
                    "ep_rew_mean": ep_rew_mean,
                    "ep_len_mean": ep_len_mean,
                }
            )
        return True

    def _on_training_end(self) -> None:
        """
        Called at the end of training to save the metrics to a CSV file.
        """
        pd.DataFrame(self.metrics).to_csv(self.log_path, index=False)
        print(f"Training metrics saved to {self.log_path}")

@dataclass
class PPOConfig:
    human_render: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 100_000
    learning_rate: float = 0.0005
    num_envs: int = 1
    num_steps: int = 1024
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 128
    norm_adv: bool = True
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOAgentSB:
    def __init__(self, env_handler: SB3EnvironmentHandler):
        self.env_handler = env_handler
        self.model = None

    def train(
        self,
        config: PPOConfig,
        verbose: int = 1,
        log_path: str = None,
    ):
        self.model = PPO(
            "MlpPolicy",
            self.env_handler.env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            # !! batch size is the number of steps times the number of environments divided by the number of minibatches
            batch_size=(config.num_steps * config.num_envs) // config.num_minibatches,
            n_steps=config.num_steps,
            n_epochs=config.update_epochs,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            clip_range=config.clip_coef,
            gae_lambda=config.gae_lambda,
            normalize_advantage=config.norm_adv,
            verbose=verbose,
            tensorboard_log=f"runs/SB3_PPO_{config.env_id}",
        )
        if log_path:
            callback = SaveTrainingMetricsCallback(log_path=log_path)
            self.model.learn(total_timesteps=config.total_timesteps, callback=callback)
        else:
            self.model.learn(total_timesteps=config.total_timesteps)

    def evaluate(self, episodes=10):
        rewards = []
        for episode in range(episodes):
            obs, _ = self.env_handler.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs)
                obs, reward, done, _, _ = self.env_handler.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        return rewards

    def save(self, path="logs/checkpoints/ppo_model"):
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}.")
        else:
            print("No trained model to save.")

    def load(self, path="logs/checkpoints/ppo_model"):
        self.model = PPO.load(path, env=self.env_handler.env)
        print(f"Model loaded from {path}.")


if __name__ == "__main__":
    config = PPOConfig()

    env_handler = SB3EnvironmentHandler(
        env_type=config.env_id,
        human_render=config.human_render,
        num_envs=config.num_envs,
    )

    agent = PPOAgentSB(
        env_handler=env_handler,
    )

    agent.train(
        config=config,
        verbose=0,
    )

    env_handler.close()
