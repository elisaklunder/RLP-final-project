from stable_baselines3 import PPO

from agents.PPO_logs_handler import SaveTrainingMetricsCallback


class PPOAgentSB:
    def __init__(self, env_handler, total_timesteps: int = 100000):
        """
        Initialize the PPOAgent with a specific environment.

        Args:
        - env_handler: Instance of Environment Handler.
        - total_timesteps: Number of timesteps for training.
        """
        self.env_handler = env_handler
        self.total_timesteps = total_timesteps
        self.model = None

    def train(
        self,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        batch_size: int = 64,
        verbose: str = 0,
        log_path: str = None,
        log_to_tensorboard: str | None = None,
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
        callback = SaveTrainingMetricsCallback(log_path=log_path)
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)

    def evaluate(self, episodes=10):
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
