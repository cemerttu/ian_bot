# ============================================================
# File: ai_filter.py
# Learns LOSSES and blocks bad trades
# ============================================================

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "ai_training.csv"
MODEL_FILE = "ai_model.pkl"
FEATURES = ["ema_gap", "rsi", "atr_ratio", "trend"]

def extract_features(df):
    i=-2
    return {
        "ema_gap": df.iloc[i]["close"]-df.iloc[i]["EMA20"],
        "rsi": df.iloc[i]["RSI"],
        "atr_ratio": df.iloc[i]["ATR"]/df.iloc[i]["ATR_MA"] if df.iloc[i]["ATR_MA"]>0 else 0,
        "trend": 1 if df.iloc[i]["EMA20"]>df.iloc[i]["EMA50"] else -1
    }

def record_trade(features, win):
    row = features.copy()
    row["win"] = int(win)
    df = pd.DataFrame([row])
    if not os.path.exists(DATA_FILE):
        df.to_csv(DATA_FILE,index=False)
    else:
        df.to_csv(DATA_FILE,mode="a",header=False,index=False)

def train_model():
    if not os.path.exists(DATA_FILE):
        return
    df = pd.read_csv(DATA_FILE)
    if len(df)<100:
        return
    X = df[FEATURES]
    y = df["win"]
    model = RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42)
    model.fit(X,y)
    joblib.dump(model,MODEL_FILE)

def ai_allow_trade(features, threshold=0.6):
    if not os.path.exists(MODEL_FILE):
        return True
    model = joblib.load(MODEL_FILE)
    prob = model.predict_proba(pd.DataFrame([features]))[0][1]
    return prob >= threshold
