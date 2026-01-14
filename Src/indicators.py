import pandas as pd

def add_technical_indicators(df):
    # exponential moving averages

    # Calculate 12-period Exponential Moving Average
    # Short-term trend indicator (reacts faster to price changes) 
    df["ema_12"] = df["Close"].ewm(span=12).mean()
    # Calculate 26-period Exponential Moving Average
    # Long-term trend indicator (smoother, slower response)
    df["ema_26"] = df["Close"].ewm(span=26).mean()