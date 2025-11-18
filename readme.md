# Hull Tactical Market Prediction
Local training environment for the Kaggle competition  
https://www.kaggle.com/competitions/hull-tactical-market-prediction

---

## 📌 Overview

This repository contains a complete local pipeline for:

- Loading Hull Tactical training and test data  
- Preprocessing + median imputation  
- Time-based splits using `date_id`  
- Training a baseline LightGBM model  
- Saving model artifacts (`.pkl`)  
- Generating Kaggle submission files  

Everything runs locally — no Kaggle Notebooks used.

---

## 📂 Project Structure

```
├── data
│   ├── raw
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed
│       ├── models
│       │   └── lgbm_baseline.pkl
│       └── train.csv
├── src
│   ├── data.py
│   └── config.py
├── scripts
│   └── train_baseline.py
└── README.md
```
