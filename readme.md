# Hull Tactical Market Prediction  
Local ML pipeline for the Kaggle competition  
https://www.kaggle.com/competitions/hull-tactical-market-prediction

---

## 📌 Project Overview

This repository contains a complete **local machine learning pipeline** (no Kaggle Notebooks) for the Hull Tactical Market Prediction competition.  

The goal is to predict  
**`market_forward_excess_returns`**  
using a wide set of market, macroeconomic, volatility, price, sentiment, and momentum features.

The project includes:
- Clean Python project structure  
- Reproducible virtual environment  
- Time-ordered training/validation  
- Multiple baseline and advanced models  
- Saved artifacts for inference  
- Separate scripts for training and (later) submissions  

---

## 📊 **Current Model Progress**

We have trained **three different models** so far using a chronological split based on `date_id`:

| Model | RMSE | Notes |
|-------|-------|-------|
| **LightGBM Baseline** | `0.00045579` | Fast, stable baseline |
| **XGBoost + Lag Features** | `0.00052525` | Slightly worse than LGBM |
| **TabNet + Lag Features (Best)** | `0.00031056` | **Best model so far (~32% lower RMSE than LGBM)** |

### 📌 Why TabNet performs best
- Learns nonlinear feature interactions  
- Handles noise + missing values well  
- Benefits from scaled target and lag features  
- Early stopping selects best epoch  

### 🏆 Current Best Model: **TabNet**  
Saved at: data/processed/models/tabnet_regressor.pkl


---

## 🛠 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/damanJ859/hull-tactical-market-prediction.git
cd hull-tactical-market-prediction
```

### 2. Create & activate virtual environment

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# or
.\.venv\Scripts\activate           # Windows

### 3. Install dependencies

pip install -r requirements.txt

### 4. Download data

Ensure your Kaggle API token is in:

- ~/.kaggle/kaggle.json (macOS/Linux)

- C:\Users\<you>\.kaggle\kaggle.json (Windows)

Then run:

```bash
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
cd data/raw && unzip "*.zip" && cd ../../
```

### 5. Train models

```bash
python -m scripts.train_baseline # LightGBM baseline
python -m scripts.train_xgboost # XGBoost with lag features
python -m scripts.train_tabnet # TabNet with lag features (best so far)
```

## ▶️ Next Planned Steps

- Add rolling-window features (returns, volatility, price).
- Train a CatBoost model as another strong tree-based baseline.
- Build ensembling: combine TabNet + LightGBM (+ XGBoost) predictions.
- Create a unified make_submission.py for Kaggle submission.
- Perform hyperparameter tuning for TabNet & LightGBM.
- Add more robust time-series cross-validation (multiple time folds).
