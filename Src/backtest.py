import matplotlib.pyplot as plt

def run_backtest(df_test, confidence):
    """
    Optional economic evaluation.
    Uses model confidence (probability of upward move) as exposure.
    """

    df_test["confidence"] = confidence

    # Transaction cost (0.1%)
    cost = 0.001

    df_test["strategy_return"] = (
        df_test["confidence"] * df_test["return"]
        - cost * df_test["confidence"].diff().abs().fillna(0)
    )

    df_test["strategy_equity"] = (1 + df_test["strategy_return"]).cumprod()
    df_test["market_equity"] = (1 + df_test["return"]).cumprod()

    # Diagnostics
    print("Average confidence:", round(df_test["confidence"].mean(), 3))

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df_test["strategy_equity"], label="Model-weighted exposure")
    plt.plot(df_test["market_equity"], label="Buy & Hold")
    plt.legend()
    plt.title("Economic Impact of Direction Predictions")
    plt.show()
