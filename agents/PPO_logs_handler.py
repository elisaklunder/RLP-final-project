import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


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
        # Check if there are new episodes
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
