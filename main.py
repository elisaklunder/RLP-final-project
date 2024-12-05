from typing import List

import numpy as np
import pandas as pd
from agents.SB3_PPO_agent import PPOAgent
from envs.environment_handler import EnvironmentHandler


def run_tuning():
    total_timesteps = 100000
    results = []

    env_handler = EnvironmentHandler(
        env_type="FlappyBird", human_render=False
    )
    agent = PPOAgent(
        env_handler=env_handler, total_timesteps=total_timesteps
    )


    agent.train(learning_rate=learning_rate, gamma=gamma, log_path=log_path)
                log_path = (
                    f"logs/tuning/gamma_{gamma}_lr_{learning_rate}_trial_{trial}.csv"
                )
                agent.train(
                    training_steps=total_timesteps,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    log_path=log_path,
                )

    env_handler.close()


    pd.DataFrame(results).to_csv("logs/tuning_metadata.csv", index=False)
    print("Tuning completed. Metadata saved to logs/tuning_metadata.csv.")


def aggregate_results(
    metadata_path: str = "logs/tuning_metadata.csv",
    output_path: str = "logs/aggregated_tuning_results.csv",
    value_columns: List[str] = None,
    hyperparameter_columns: List[str] = None,
):
    """
    Analyze tuning results and aggregate data dynamically for any set of hyperparameters and metrics.

    Args:
    - metadata_path: Path to the metadata CSV containing hyperparameters and log paths.
    - output_path: Path to save the aggregated results as a CSV file.
    - value_columns: List of columns to aggregate (e.g., ["ep_rew_mean", "ep_len_mean"]).
      If None, all columns except `timesteps` and hyperparameters will be aggregated.
    - hyperparameter_columns: List of hyperparameter columns to group by.
      If None, all columns except `log_path` and `timesteps` will be considered hyperparameters.

    Returns:
    - final_results: DataFrame with aggregated results.
    """
    metadata = pd.read_csv(metadata_path)
    aggregated_data = []

    if hyperparameter_columns is None:
        hyperparameter_columns = [
            col for col in metadata.columns if col not in ["log_path"]
        ]

    for hyperparams, group in metadata.groupby(hyperparameter_columns):
        combined_data = []
        for _, row in group.iterrows():
            trial_data = pd.read_csv(row["log_path"])
            combined_data.append(trial_data)

        combined_df = pd.concat(combined_data)

        if value_columns is None:
            value_columns = [
                col for col in combined_df.columns if col not in ["timesteps"]
            ]

        agg_funcs = {}
        for value_col in value_columns:
            agg_funcs[f"mean_{value_col}"] = (value_col, "mean")
            agg_funcs[f"sem_{value_col}"] = (
                value_col,
                lambda x: x.std() / np.sqrt(len(x)),
            )

        aggregated_df = combined_df.groupby("timesteps").agg(**agg_funcs).reset_index()

        for col, value in zip(hyperparameter_columns, hyperparams):
            aggregated_df[col] = value

        aggregated_data.append(aggregated_df)

    final_results = pd.concat(aggregated_data)
    final_results.to_csv(output_path, index=False)

    return final_results


if __name__ == "__main__":
    # run_tuning()
    # final_results = aggregate_results()
    # plot_training_metrics_with_sem(final_results, figure_id="PPO_agents")
    # results = pd.read_csv("logs/aggregated_tuning_results.csv")
    # calculate_mean_and_se(
    #     results,
    #     value_columns=["mean_ep_rew", "mean_ep_len"],
    #     hyperparameter_columns=["learning_rate", "gamma"],
    # )

    env_handler = EnvironmentHandler(env_type="FlappyBird", human_render=True)
    agent = PPOAgent(env_handler=env_handler)
    agent.train()
