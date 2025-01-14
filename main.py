import numpy as np
import pandas as pd

from agents.SB3_ppo import PPOAgentSB
from envs.environment_handler import EnvironmentHandler
from utils.plot import plot_training_metrics_with_sem


def run_tuning():
    gammas = [0.95, 0.99]
    learning_rates = [0.0005, 0.005]
    n_trials = 5
    total_timesteps = 100000

    results = []

    for gamma in gammas:
        for learning_rate in learning_rates:
            for trial in range(1, n_trials + 1):
                print(
                    f"Running trial {trial} for gamma={gamma}, learning_rate={learning_rate}"
                )

                env_handler = EnvironmentHandler(
                    env_type="FlappyBird", human_render=False
                )
                agent = PPOAgentSB(
                    env_handler=env_handler, total_timesteps=total_timesteps
                )

                log_path = (
                    f"logs/tuning/gamma_{gamma}_lr_{learning_rate}_trial_{trial}.csv"
                )
                agent.train(learning_rate=learning_rate, gamma=gamma, log_path=log_path)

                env_handler.close()

                results.append(
                    {
                        "gamma": gamma,
                        "learning_rate": learning_rate,
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

    for (gamma, lr), group in metadata.groupby(["gamma", "learning_rate"]):
        combined_data = []

        for _, row in group.iterrows():
            # Load metrics for each trial
            trial_data = pd.read_csv(row["log_path"])
            combined_data.append(trial_data)

        # Combine data for the same hyperparameter combination
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

        # Add hyperparameters for reference
        df["gamma"] = gamma
        df["learning_rate"] = lr
        aggregated_data.append(df)

    # Combine all results into a single DataFrame
    final_results = pd.concat(aggregated_data)
    final_results.to_csv("logs/aggregated_tuning_results.csv", index=False)

    # Generate plots
    plot_training_metrics_with_sem(final_results, figure_id="PPO_agents")


if __name__ == "__main__":
    # run_tuning()
    #analyze_tuning_results()
    
