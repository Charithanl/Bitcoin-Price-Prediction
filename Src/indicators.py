import pandas as pd

def add_technical_indicators(df):
    # exponential moving averages

    # Calculate 12-period Exponential Moving Average
    # Short-term trend indicator (reacts faster to price changes) 
    df["ema_12"] = df["Close"].ewm(span=12).mean()
    # Calculate 26-period Exponential Moving Average
    # Long-term trend indicator (smoother, slower response)
    df["ema_26"] = df["Close"].ewm(span=26).mean()

    # MACD (Moving Average Convergence Divergence)
    # MACD line = difference between short-term EMA and long-term EMA
    # Shows momentum and trend direction
    df["macd"] = df["ema_12"] - df["ema_26"]

    # MACD signal line
    # 9-period EMA of the MACD line, used to generate buy/sell signals
    df["macd_signal"] = df["macd"].ewm(span=9).mean()