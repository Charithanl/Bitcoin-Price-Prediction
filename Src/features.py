def build_features(df):
    df["open_close"] = df["Open"] - df["Close"]
    df["low_high"] = df["Low"] - df["High"]
    df["return"] = df["Close"].pct_change()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)