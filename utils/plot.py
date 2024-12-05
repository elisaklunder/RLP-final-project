import matplotlib.pyplot as plt
import pandas as pd


def plot_training_metrics_with_sem(results: pd.DataFrame, figure_id: str):
    """
    Plot training metrics with mean and SEM for all hyperparameter combinations on the same figure.

    Args:
    - results: DataFrame containing aggregated results (mean and SEM).
    - figure_id: how to identify the figure.
    """

    # Get hyperparameter columns (excluding timesteps, means and SEMs)
    hyperparameter_cols = [col for col in results.columns if col not in 
                          ['timesteps', 'mean_ep_rew', 'sem_ep_rew', 
                           'mean_ep_len', 'sem_ep_len']]

    plt.figure(figsize=(10, 6))
    for _, group in results.groupby(hyperparameter_cols):
        # Create label from all hyperparameters
        label_parts = []
        for col in hyperparameter_cols:
            # Format the value based on type
            val = group[col].iloc[0]
            if isinstance(val, float):
                if col == 'gamma':
                    label_parts.append(f"γ={val}")
                else:
                    label_parts.append(f"{col}={val:.4f}")
            else:
                label_parts.append(f"{col}={val}")
        label = ", ".join(label_parts)
        
        plt.plot(group["timesteps"], group["mean_ep_rew"], label=label)
        plt.fill_between(
            group["timesteps"],
            group["mean_ep_rew"] - group["sem_ep_rew"],
            group["mean_ep_rew"] + group["sem_ep_rew"],
            alpha=0.2,
        )
    plt.title("Mean Episode Reward with SEM")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.legend(title="Hyperparameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()

    # Ensure the directory exists before saving the plot
    import os
    os.makedirs("plots", exist_ok=True)

    plt.savefig(f"plots/{figure_id}_reward_comparison.png", bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    for _, group in results.groupby(hyperparameter_cols):
        # Create label from all hyperparameters
        label_parts = []
        for col in hyperparameter_cols:
            # Format the value based on type
            val = group[col].iloc[0]
            if isinstance(val, float):
                if col == 'gamma':
                    label_parts.append(f"γ={val}")
                else:
                    label_parts.append(f"{col}={val:.4f}")
            else:
                label_parts.append(f"{col}={val}")
        label = ", ".join(label_parts)
        
        plt.plot(group["timesteps"], group["mean_ep_len"], label=label)
        plt.fill_between(
            group["timesteps"],
            group["mean_ep_len"] - group["sem_ep_len"],
            group["mean_ep_len"] + group["sem_ep_len"],
            alpha=0.2,
        )
    plt.title("Mean Episode Length with SEM")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Length")
    plt.legend(title="Hyperparameters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()

    # Ensure the directory exists before saving the plot
    os.makedirs("plots", exist_ok=True)

    plt.savefig(f"plots/{figure_id}_episode_length_comparison.png", bbox_inches='tight')
    plt.close()

    print("Plotting succesfull.")
