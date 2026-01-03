# ============================================================
# File: ai_filter.py
# Learns from PAST trade outcomes to block likely losing trades
# Uses machine learning (Random Forest) — a valid AI technique
# ============================================================

import pandas as pd
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
DATA_FILE = "ai_training.csv"
MODEL_FILE = "ai_model.pkl"
FEATURES = ["ema_gap", "rsi", "atr_ratio", "trend"]
MIN_SAMPLES_TO_TRAIN = 150  # Increased to reduce overfitting
MIN_TEST_SIZE = 30

def extract_features(df):
    """
    Extract features from the **latest completed bar** (index -1).
    Assumes this function is called at the close of a new bar,
    and the trade decision would be made using this bar's data.
    """
    if len(df) < 2:
        raise ValueError("DataFrame must have at least 2 rows to extract features.")
    
    i = -1  # Use the latest completed bar
    row = df.iloc[i]
    
    ema_gap = row["close"] - row["EMA20"]
    rsi = row["RSI"]
    atr_ma = row["ATR_MA"]
    atr_ratio = row["ATR"] / atr_ma if atr_ma > 1e-8 else 0.0
    trend = 1 if row["EMA20"] > row["EMA50"] else -1

    return {
        "ema_gap": float(ema_gap),
        "rsi": float(rsi),
        "atr_ratio": float(atr_ratio),
        "trend": int(trend)
    }

def log_trade_candidate(features):
    """
    Logs a trade candidate **before knowing the outcome**.
    This row will later be updated with 'win' once outcome is known.
    We store a temporary row with win = -1 (unknown).
    """
    row = features.copy()
    row["win"] = -1  # -1 means outcome not yet known
    df_row = pd.DataFrame([row])
    
    if not os.path.exists(DATA_FILE):
        df_row.to_csv(DATA_FILE, index=False)
    else:
        df_row.to_csv(DATA_FILE, mode="a", header=False, index=False)

def update_last_trade_outcome(win: bool):
    """
    Updates the **most recent** trade candidate with its actual outcome.
    Call this AFTER the trade is closed and result is known.
    """
    if not os.path.exists(DATA_FILE):
        print("Warning: No training file to update outcome.")
        return

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        return

    # Find last row with win == -1 (pending outcome)
    pending_idx = df[df['win'] == -1].index
    if len(pending_idx) == 0:
        print("Warning: No pending trade to update.")
        return

    last_pending = pending_idx[-1]
    df.at[last_pending, 'win'] = int(win)
    df.to_csv(DATA_FILE, index=False)

def record_trade(features, win):
    """
    Records a completed trade into the training CSV with known outcome.
    `features` should be a dict matching the FEATURES keys.
    `win` is a boolean or int (True/1 for win, False/0 for loss).
    """
    row = features.copy()
    row['win'] = int(bool(win))
    df_row = pd.DataFrame([row])
    if not os.path.exists(DATA_FILE):
        df_row.to_csv(DATA_FILE, index=False)
    else:
        df_row.to_csv(DATA_FILE, mode='a', header=False, index=False)

def train_model():
    """
    Trains the model only on rows with known outcomes (win == 0 or 1).
    Uses train/test split and class balancing.
    """
    if not os.path.exists(DATA_FILE):
        return

    df = pd.read_csv(DATA_FILE)
    # Keep only resolved trades
    df = df[df['win'] != -1].copy()
    
    if len(df) < MIN_SAMPLES_TO_TRAIN:
        print(f"AI: Not enough resolved trades ({len(df)} < {MIN_SAMPLES_TO_TRAIN}). Skipping training.")
        return

    X = df[FEATURES]
    y = df["win"].astype(int)

    if len(np.unique(y)) < 2:
        print("AI: Only one class present. Skipping training.")
        return

    # Ensure enough samples for test set
    test_size = max(0.2, MIN_TEST_SIZE / len(df))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if stratify fails (e.g., too few samples per class)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Optional: log performance
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"AI: Model trained on {len(df)} samples. Test accuracy: {acc:.2%}")

    joblib.dump(model, MODEL_FILE)

def ai_allow_trade(features, threshold=0.55):
    """
    Returns True if model predicts win probability >= threshold.
    Returns True if model not available (fail-open).
    """
    if not os.path.exists(MODEL_FILE):
        return True  # Allow trade if no model (e.g., early stage)

    try:
        model = joblib.load(MODEL_FILE)
        X = pd.DataFrame([features])
        prob_win = model.predict_proba(X)[0][1]  # Prob of class 1 (win)
        return prob_win >= threshold
    except Exception as e:
        print(f"AI: Error during prediction: {e}. Allowing trade as fallback.")
        return True