import pandas as pd
import gymnasium as gym
import gym_trading_env
from dataclasses import dataclass
from datetime import datetime
from stable_baselines3 import PPO
# Available in the github repo : examples/data/BTC_USD-Hourly.csv

total_timesteps: int = 100_000
learning_rate: float = 0.0005
num_envs: int = 1
num_steps: int = 1024
gamma: float = 0.95
gae_lambda: float = 0.95
num_minibatches: int = 4
update_epochs: int = 128
norm_adv: bool = True
clip_coef: float = 0.2
vf_coef: float = 0.5
max_grad_norm: float = 0.5
def download_dataset():
    url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
    df = pd.read_csv(url, parse_dates=["date"], index_col= "date")
    df.sort_index(inplace= True)
    df.dropna(inplace= True)
    df.drop_duplicates(inplace=True)
    return df


def make_featueres(df):
    # df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"

    # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change()

    # Create the feature : open[t] / close[t]
    df["feature_open"] = df["open"]/df["close"]

    # Create the feature : high[t] / close[t]
    df["feature_high"] = df["high"]/df["close"]

    # Create the feature : low[t] / close[t]
    df["feature_low"] = df["low"]/df["close"]

    # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

    df.dropna(inplace= True) # Clean again !
    # Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"

    return df



df = download_dataset()
make_featueres(df)
env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=learning_rate,
    gamma=gamma,
    # !! batch size is the number of steps times the number of environments divided by the number of minibatches
    batch_size=(num_steps * num_envs) // num_minibatches,
    n_steps=num_steps,
    n_epochs=update_epochs,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    clip_range=clip_coef,
    gae_lambda=gae_lambda,
    normalize_advantage=norm_adv,
    tensorboard_log=f"runs/SB3_PPO_trading_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)