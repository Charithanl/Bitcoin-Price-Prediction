import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src.indicators import add_technical_indicators
from src.features import build_features
from src.model import train_model
from src.backtest import run_backtest

# Load data
df = pd.read_csv("data/bitcoin.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df.drop(columns=["Adj Close"], inplace=True)

# Feature engineering
df = add_technical_indicators(df)
df, feature_cols = build_features(df)

# Time-based split
split = int(len(df) * 0.7)

X = df[feature_cols]
y = df["target"]

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model, scaler = train_model(X_train, y_train)

# Predict probabilities
X_test_scaled = scaler.transform(X_test)
probs = model.predict_proba(X_test_scaled)[:, 1]

# Classification evaluation (direction prediction)
for t in [0.45, 0.48, 0.50]:
    preds = (probs > t).astype(int)
    print(f"\nThreshold: {t}")
    print(classification_report(y_test, preds))
preds = (probs > 0.48).astype(int)
print("\nOverall Performance at 0.48 Threshold:")

print("ROC-AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test_scaled,
    y_test,
    cmap="Blues"
)
plt.show()

print("Positive prediction rate:", preds.mean())

# Optional economic evaluation (confidence-weighted)
run_backtest(df.iloc[split:].copy(), probs)
