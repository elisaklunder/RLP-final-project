from stable_baselines3 import PPO

from agents.PPO_logs_handler import SaveTrainingMetricsCallback


class PPOAgent:
    def __init__(self, env_handler):
        """
        Initialize the PPOAgent with a specific environment.

        Args:
        - env_handler: Instance of Environment Handler.
        - total_timesteps: Number of timesteps for training.
        """
        self.env_handler = env_handler
        self.model = None

    def train(
        self,
        training_steps=100000,
        learning_rate: float = 0.0005,
        gamma: float = 0.99,
        batch_size: int = 64,
        verbose: str = 0,
        log_path: str = None,
        log_to_tensorboard: str | None = None,
        progress_bar: bool = False,
    ):
        """
        Train the PPO agent using the environment handler and log to TensorBoard.
        """
        self.model = PPO(
            "MlpPolicy",
            self.env_handler.env,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            verbose=verbose,
            tensorboard_log=log_to_tensorboard,
        )
        if log_path:
            log_path = SaveTrainingMetricsCallback(log_path=log_path)
        self.model.learn(
            total_timesteps=training_steps, callback=log_path, progress_bar=progress_bar
        )

    def evaluate(self, episodes=100):
        """
        Evaluate the PPO agent.
        """
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
        """
        Save the trained PPO model to the specified path.
        """
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}.")
        else:
            print("No trained model to save.")

    def load(self, path="logs/checkpoints/ppo_model"):
        """
        Load a PPO model from the specified path.
        """
        self.model = PPO.load(path, env=self.env_handler.env)
        print(f"Model loaded from {path}.")
