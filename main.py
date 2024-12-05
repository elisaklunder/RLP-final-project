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

    env_handler.close()


    pd.DataFrame(results).to_csv("logs/tuning_metadata.csv", index=False)
    print("Tuning completed. Metadata saved to logs/tuning_metadata.csv.")


if __name__ == "__main__":
    run_tuning()
