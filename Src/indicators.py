def add_technical_indicators(df):
    df["ema_12"] = df["Close"].ewm(span=12).mean()
    df["ema_26"] = df["Close"].ewm(span=26).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + rs))

    ma = df["Close"].rolling(20)
    df["bb_upper"] = ma.mean() + 2 * ma.std()
    df["bb_lower"] = ma.mean() - 2 * ma.std()

    return df   # ← THIS LINE WAS MISSING
