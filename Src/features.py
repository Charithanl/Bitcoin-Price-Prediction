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

    # Target variable:
    # 1 if next day's Close price is higher than today's Close
    # 0 otherwise
    # shift(-1) moves future Close price to current row
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove rows with NaN values created by pct_change() and shift()
    # Machine learning models cannot work with missing values
    df.dropna(inplace=True)

    # List of feature columns to be used for model training
    features = [
        "open_close",   # Intraday price movement
        "low_high",     # Volatility indicator
        "return",       # Price return
        "rsi",          # Relative Strength Index (momentum)
        "macd",         # Moving Average Convergence Divergence
        "macd_signal",  # MACD signal line
        "ema_12",       # Short-term exponential moving average
        "ema_26",       # Long-term exponential moving average
        "bb_upper",     # Upper Bollinger Band (volatility)
        "bb_lower"      # Lower Bollinger Band (volatility)
    ]

    # Return the updated DataFrame and feature list
    return df, features
