def build_features(df):
    # Feature 1: Difference between Open and Close price
    # Helps identify bullish or bearish candles
    df["open_close"] = df["Open"] - df["Close"]
    # Feature 2: Difference between Low and High price
    # Measures price volatility within a time period
    df["low_high"] = df["Low"] - df["High"]
    # Feature 3: Percentage return of Close price
    # Shows relative price movement compared to previous period
    df["return"] = df["Close"].pct_change()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)