# Bitcoin Price Direction Prediction using Machine Learning

## Overview

* This project explores whether **historical Bitcoin price data (OHLC)** contains enough information to predict **future price direction** using machine learning.
* The task is framed as a **binary classification problem**:

  * `1` → price goes up after a fixed future horizon
  * `0` → price does not go up
* The goal is **analysis and learning**, not building a live trading system.

---

## Key Objectives

* Build a **time-series–correct ML pipeline** (no data leakage)
* Engineer features from historical price data
* Evaluate model performance using **appropriate metrics**
* Analyze **limitations and failure cases** honestly

---

## Dataset

* Historical Bitcoin OHLC data
* Columns used:

  * Open
  * High
  * Low
  * Close
* Data is sorted chronologically and split using **past → future** logic

---

## Feature Engineering

Derived features include:

* Open–Close difference
* Low–High difference
* Percentage returns
* Technical indicators:

  * RSI (Relative Strength Index)
  * MACD & MACD signal
  * EMA (12, 26)
  * Bollinger Bands (upper & lower)

Target variable:

* Binary label indicating whether the **future closing price (weekly horizon)** is higher than the current close

---

## Model

* **Logistic Regression**
* Reasons for choosing this model:

  * Strong, interpretable baseline
  * Stable under weak signal conditions
  * Suitable for probability-based evaluation
* Feature scaling performed using `StandardScaler`

  * Fitted **only on training data** to avoid leakage

---

## Train–Test Strategy

* **Time-based split**

  * First 70% → training
  * Remaining 30% → testing
* No random shuffling is used
* This setup reflects **real-world prediction constraints**

---

## Evaluation Metrics

* **Primary metric:** ROC-AUC

  * Measures ranking quality of predicted probabilities
* **Secondary metrics:**

  * Precision, recall, F1-score
  * Confusion matrix (for analysis, not optimization)

Why not accuracy?

* Accuracy is misleading for noisy financial time series
* ROC-AUC is more appropriate for probabilistic classifiers

---

## Results (Summary)

* ROC-AUC ≈ **0.59**
* The model:

  * Is **slightly better than random**
  * Is **conservative** (low recall for upward moves)
* Binary predictions underperform buy-and-hold strategies

These results indicate:

* Weak but real predictive signal
* Strong limitations when using **price-only data**

---

## Economic Evaluation (Optional)

* A simple backtest is included **only to assess practical relevance**
* Uses **model confidence (probabilities)**, not hard trading rules
* No claim of profitability is made

---

## Key Insights

* Short-horizon Bitcoin price direction is **hard to predict**
* Technical indicators do not add new information beyond price
* Correct ML pipelines often reveal **limits of predictability**, not high accuracy
* Poor-looking metrics can be **more honest** than inflated ones

---

## Limitations

* Uses only OHLC price data
* No volume, sentiment, macro, or on-chain data
* Not suitable for live trading or deployment
* Results should not be interpreted as financial advice

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Structure

```
bitcoin_price_prediction/
│
├── data/
│   └── bitcoin.csv
│
├── src/
│   ├── indicators.py
│   ├── features.py
│   ├── model.py
│   └── backtest.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

1. Create and activate a virtual environment
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the project:

   ```
   python main.py
   ```

---

## Conclusion

This project demonstrates that:

* Correct machine learning practices are more important than impressive metrics
* Financial time series often have **low signal-to-noise ratios**
* Understanding *why* a model fails can be more valuable than forcing success

---

## Disclaimer

This project is for **educational and analytical purposes only**.
It does **not** constitute financial or investment advice.

