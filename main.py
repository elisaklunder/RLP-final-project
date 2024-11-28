import os
import pandas as pd
import matplotlib.pyplot as plt
from agents.PPO_agent import PPOAgent
from envs.environment_handler import EnvironmentHandler


def plot_rewards(rewards, filename="plots/rewards_plot.png"):
    """
    Plot the rewards and save the plot to a file.
    """
    plt.figure()
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Rewards plot saved to {filename}.")

if __name__ == "__main__":

    env_handler = EnvironmentHandler(env_type="FlappyBird", human_render=True)
    agent = PPOAgent(env_handler=env_handler, total_timesteps=100000)

    # agent.train(learning_rate=0.0003, gamma=0.99, batch_size=64)
    # agent.save("logs/checkpoints/ppo_model")

    # Save and plot rewards
    agent.load("logs/checkpoints/ppo_model")
    rewards = agent.evaluate(episodes=10)
    rewards_df = pd.DataFrame({"episode": list(range(1, len(rewards) + 1)), "reward": rewards})
    rewards_df.to_csv("logs/rewards.csv", index=False)
    plot_rewards(rewards, filename="plots/rewards_plot.png")

    env_handler.close()
