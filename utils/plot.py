import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics_with_sem(results: pd.DataFrame, figure_id: str):
    """
    Plot training metrics with mean and SEM for all hyperparameter combinations on the same figure.

    Args:
    - results: DataFrame containing aggregated results (mean and SEM).
    - figure_id: how to identify the figure. 
    """

    plt.figure(figsize=(10, 6))
    for (gamma, lr), group in results.groupby(["gamma", "learning_rate"]):
        plt.plot(group["timesteps"], group["mean_ep_rew"], label=f"γ={gamma}, LR={lr}")
        plt.fill_between(
            group["timesteps"],
            group["mean_ep_rew"] - group["sem_ep_rew"],
            group["mean_ep_rew"] + group["sem_ep_rew"],
            alpha=0.2,
        )
    plt.rcParams.update({'font.size': 14})
    plt.title("Mean Episode Reward with SEM")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend(title="Hyperparameters")
    plt.grid()

    # Ensure the directory exists before saving the plot
    import os
    os.makedirs("plots", exist_ok=True)

    plt.savefig(f"plots/{figure_id}_reward_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    for (gamma, lr), group in results.groupby(["gamma", "learning_rate"]):
        plt.plot(group["timesteps"], group["mean_ep_len"], label=f"γ={gamma}, LR={lr}")
        plt.fill_between(
            group["timesteps"],
            group["mean_ep_len"] - group["sem_ep_len"],
            group["mean_ep_len"] + group["sem_ep_len"],
            alpha=0.2,
        )
    plt.title("Mean Episode Length with SEM")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.legend(title="Hyperparameters")
    plt.grid()

    # Ensure the directory exists before saving the plot
    os.makedirs("plots", exist_ok=True)

    plt.savefig(f"plots/{figure_id}_episode_length_comparison.png")
    plt.close()
