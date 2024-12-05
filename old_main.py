import numpy as np
import pandas as pd

from agents.SB3_PPO_agent import PPOAgent
from envs.environment_handler import EnvironmentHandler
from utils.plot import plot_training_metrics_with_sem


def run_tuning():
    gamma = 0.095
    learning_rate = 0.0005
    batch_sizes = [32, 64, 128]
    epochs_per_update = [4, 10, 16]
    buffer_size = [1024, 2048]
    n_trials = 3
    total_timesteps = 1000000

    results = []

    for batch_size in batch_sizes:
        for k in epochs_per_update:
            for rollout_buffer_size in buffer_size:
                for trial in range(1, n_trials + 1):
                    print(
                        f"Running trial {trial} for batch_size={batch_size}, k={k}, buffer_size={rollout_buffer_size}"
                    )

                    env_handler = EnvironmentHandler(
                        env_type="FlappyBird", human_render=False
                    )
                    agent = PPOAgent(env_handler=env_handler)

                    log_path = (
                        f"logs/tuning/batch_size_{batch_size}_epochs_update_{k}_buffer_size_{rollout_buffer_size}_trial_{trial}.csv"
                    )
                    agent.train(training_steps=total_timesteps, 
                                learning_rate=learning_rate, 
                                rollout_buffer_size=rollout_buffer_size,
                                batch_size=batch_size,
                                epochs_per_update=k, 
                                gamma=gamma,
                                log_path=log_path, 
                                log_to_tensorboard="logs/tensorboard")

                    env_handler.close()

                    results.append(
                        {
                            "B": batch_size,
                            "k": k,
                            "D": rollout_buffer_size,
                            "trial": trial,
                            "log_path": log_path,
                        }
                    )

    pd.DataFrame(results).to_csv("logs/tuning_metadata.csv", index=False)
    print("Tuning completed. Metadata saved to logs/tuning_metadata.csv.")


def analyze_tuning_results():
    """
    Analyze tuning results and generate plots.
    """
    # Load tuning metadata
    metadata = pd.read_csv("logs/tuning_metadata.csv")
    aggregated_data = []
    hyperparameter_cols = [col for col in metadata.columns if col not in ["trial", "log_path"]]

    # Group by all hyperparameter columns
    for hyperparams, group in metadata.groupby(hyperparameter_cols):
        combined_data = []

        for _, row in group.iterrows():
            trial_data = pd.read_csv(row["log_path"])
            combined_data.append(trial_data)

        df = (
            pd.concat(combined_data)
            .groupby("timesteps")
            .agg(
                mean_ep_rew=("ep_rew_mean", "mean"),
                sem_ep_rew=("ep_rew_mean", lambda x: x.std() / np.sqrt(len(x))),
                mean_ep_len=("ep_len_mean", "mean"),
                sem_ep_len=("ep_len_mean", lambda x: x.std() / np.sqrt(len(x))),
            )
            .reset_index()
        )

        for col, value in zip(hyperparameter_cols, hyperparams):
            df[col] = value

        aggregated_data.append(df)

    final_results = pd.concat(aggregated_data)
    final_results.to_csv("logs/aggregated_tuning_results.csv", index=False)
    plot_training_metrics_with_sem(final_results, figure_id="PPO_agents")


if __name__ == "__main__":
    run_tuning()
    analyze_tuning_results()
