from datetime import datetime

import gym_trading_env  # noqa
import gymnasium as gym
import pandas as pd
from stable_baselines3 import PPO

total_timesteps: int = 1_000_000
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
    df = pd.read_csv(url, parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def make_features(df):
    # 1. ( close[t] - close[t-1] )/ close[t-1]
    df["feature_close"] = df["close"].pct_change()

    # 2. open[t] / close[t]
    df["feature_open"] = df["open"] / df["close"]

    # 3. high[t] / close[t]
    df["feature_high"] = df["high"] / df["close"]

    # 4. low[t] / close[t]
    df["feature_low"] = df["low"] / df["close"]

    # 5. volume[t] / max(*volume[t-7*24:t+1])
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()

    # -------------------------------------------------------------------------
    # 6. Moving Average (MA) - e.g. 24-period (1 day) simple moving average
    df["feature_ma_24"] = df["close"].rolling(window=24).mean() / df["close"]

    # 7. Exponential Moving Average (EMA) - e.g. 72-period (3 days)
    df["feature_ema_72"] = df["close"].ewm(span=72, adjust=False).mean() / df["close"]

    # -------------------------------------------------------------------------
    # 8. Rolling Volatility - e.g. 24-period standard deviation
    df["feature_volatility_24"] = df["close"].pct_change().rolling(window=24).std()

    # -------------------------------------------------------------------------
    # 9. Relative Strength Index (RSI)
    #    RSI = 100 - (100 / (1 + RS)),
    #    where RS = (average gain over N periods) / (average loss over N periods).
    #    We'll pick a 14-period RSI for example.

    window_rsi = 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use exponential moving average for RSI
    alpha = 1.0 / window_rsi
    avg_gain = gain.ewm(alpha=alpha, min_periods=window_rsi).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=window_rsi).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # small epsilon to avoid division by zero
    df["feature_rsi"] = 100 - (100 / (1 + rs))

    # Normalize RSI a bit to keep everything in a somewhat consistent scale
    df["feature_rsi"] = df["feature_rsi"] / 100.0  # [0,1] range

    # -------------------------------------------------------------------------
    # 10. MACD (Moving Average Convergence Divergence)
    #     MACD = EMA(12) - EMA(26)
    #     We'll store the difference normalized by the price to keep scale smaller.

    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["feature_macd"] = (ema_fast - ema_slow) / df["close"]

    # -------------------------------------------------------------------------
    # 11. Bollinger Bands
    #     Typically: middle band = MA(20),
    #     upper band = MA(20) + 2 * std(20),
    #     lower band = MA(20) - 2 * std(20).
    #     We'll store the band width and band position as features.

    window_bb = 20
    ma_bb = df["close"].rolling(window=window_bb).mean()
    std_bb = df["close"].rolling(window=window_bb).std()

    df["feature_bb_width"] = (2 * std_bb) / ma_bb  # how wide the band is
    df["feature_bb_position"] = (df["close"] - ma_bb) / (
        2 * std_bb
    )  # position relative to bands

    # -------------------------------------------------------------------------
    # Finally, drop any rows with NaN from newly created features
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":
    df = download_dataset()
    df = make_features(df)

    env = gym.make(
        "TradingEnv",
        name="BTCUSD",
        df=df,
        positions=[-1, -0.5, 0, 0.5, 1],  # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate=0.0003
        / 100,  # 0.0003% per timestep (one timestep = 1h here)
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
    model.learn(total_timesteps=total_timesteps)
    
    # for i in range(100):
    #     done, truncated = False, False
    #     observation, info = env.reset()

    #     while not done and not truncated:
    #         position_index, _ = model.predict(observation)
    #         observation, reward, done, truncated, info = env.step(position_index)

    #     env.unwrapped.save_for_render(dir="render_logs")
