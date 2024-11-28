from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd

class SaveTrainingMetricsCallback(BaseCallback):
    """
    Custom callback for saving training metrics to a CSV file and generating plots.
    """
    def __init__(self, log_path="logs/training_metrics.csv", verbose=0):
        super(SaveTrainingMetricsCallback, self).__init__(verbose)
        self.log_path = log_path
        self.metrics = []

    def _on_step(self) -> bool:
        self.metrics.append({
            "timesteps": self.num_timesteps,
            "ep_rew_mean": self.locals.get("runner").reward_buffer.mean(),
            "ep_len_mean": self.locals.get("runner").episode_lengths.mean()
        })
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(self.metrics).to_csv(self.log_path, index=False)
