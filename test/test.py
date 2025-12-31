import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from train.train import StockTradingEnv, merge_data


def backtest():
    data = merge_data()
    env = DummyVecEnv([lambda: StockTradingEnv(data, mode="test")])

    model = PPO.load(
        "lstm_ppo_trader.zip",
        env=env,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )

    obs = env.reset()
    done = False

    portfolio_values = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        val = info[0]["portfolio_value"]
        portfolio_values.append(val)

    portfolio_values = np.array(portfolio_values)
    initial_value = 1_000_000
    final_value = portfolio_values[-1]
    cumulative_return = (final_value - initial_value) / initial_value
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

    peaks = pd.Series(portfolio_values).cummax()
    drawdowns = (portfolio_values - peaks) / peaks
    max_drawdown = drawdowns.min()

    print("-" * 30)
    print(f"Initial Value:   ${initial_value:,.2f}")
    print(f"Final Value:     ${final_value:,.2f}")
    print(f"Cumulative Ret:  {cumulative_return*100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe_ratio:.4f}")
    print(f"Max Drawdown:    {max_drawdown*100:.2f}%")
    print("-" * 30)

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label="LSTM-PPO Agent")
    plt.title(f"Backtest Performance (2019-2023)\nReturn: {cumulative_return*100:.2f}%")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("backtest_result.png")
    print("Plot saved to backtest_result.png")


if __name__ == "__main__":
    backtest()
