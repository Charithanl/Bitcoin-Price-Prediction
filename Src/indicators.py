import pandas as pd

def add_technical_indicators(df):
    df["ema_12"] = df["Close"].ewm(span=12).mean()
    df["ema_26"] = df["Close"].ewm(span=26).mean()