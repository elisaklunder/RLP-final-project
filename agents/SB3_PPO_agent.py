import flappy_bird_gymnasium  # Add this at the top of the file
from stable_baselines3 import PPO
from agents.PPO_logs_handler import SaveTrainingMetricsCallback
from envs.environment_handler import EnvironmentHandler
from dataclasses import dataclass

@dataclass
class PPOConfig:
    human_render: bool = False
    env_id: str = "FlappyBird-v0"
    total_timesteps: int = 100_000
    learning_rate: float = 0.0005
    num_envs: int = 4
    num_steps: int = 100
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class PPOAgentSB:
    def __init__(self, env_handler):
        self.env_handler = env_handler
        self.model = None

    def train(
        self,
        config: PPOConfig,
        verbose: int = 1,
        log_path: str = None,
        log_to_tensorboard: str = "logs/tensorboard",
    ):
        log_to_tensorboard = f"{log_to_tensorboard}/{self.env_handler.env_type}"
        self.model = PPO(
            "MlpPolicy",
            self.env_handler.env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,

            # !! batch size is the number of steps times the number of environments divided by the number of minibatches
            batch_size=(config.num_steps * config.num_envs)// config.num_minibatches, 

            n_steps=config.num_steps,
            n_epochs=config.update_epochs,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            clip_range=config.clip_coef,
            gae_lambda=config.gae_lambda,         
            normalize_advantage=config.norm_adv,
            verbose=verbose,
            tensorboard_log=log_to_tensorboard,
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
    env_handler = EnvironmentHandler(
        env_type=config.env_id, 
        human_render=config.human_render, 
        num_envs=config.num_envs
    )

    agent = PPOAgentSB(
        env_handler=env_handler, 
    )

    agent.train(
        config=config,
        verbose=1,
    )

    env_handler.close()
