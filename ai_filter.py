# ============================================================
# File: ai_filter.py
# Advanced AI Filter with Contextual Awareness
# ============================================================

import pandas as pd
import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
DATA_FILE = "ai_training.csv"
MODEL_FILE = "ai_model.pkl"

# Expanded Features for better context
FEATURES = [
    "ema_gap", 
    "rsi", 
    "atr_ratio", 
    "trend", 
    "hour",           # Time Context
    "vol_index",      # Volatility Context
    "ema_slope"       # Trend Strength
]

MIN_SAMPLES_TO_TRAIN = 200  # Increased for better stability
MIN_TEST_SIZE = 40

def extract_features(df):
    """
    Extracts advanced features from the latest CLOSED candle.
    """
    if len(df) < 50:
        raise ValueError("DataFrame needs more rows for stable feature extraction.")
    
    # Use index -2 (the last fully closed candle) to avoid data leakage
    row = df.iloc[-2]
    prev_row = df.iloc[-3]
    
    # 1. Basic Technicals
    ema_gap = row["close"] - row["EMA20"]
    rsi = row["RSI"]
    atr_ma = row["ATR_MA"]
    atr_ratio = row["ATR"] / atr_ma if atr_ma != 0 else 1
    
    # 2. Trend Direction (1 for Up, 0 for Down)
    trend = 1 if row["EMA20"] > row["EMA50"] else 0
    
    # 3. Time Context (Hour of the day in UTC)
    hour = datetime.utcnow().hour
    
    # 4. Volatility Index (Current Volatility vs Long-term Volatility)
    # Checks if market is currently "quiet" or "explosive"
    long_term_atr = df["ATR"].rolling(50).mean().iloc[-1]
    vol_index = row["ATR"] / long_term_atr if long_term_atr != 0 else 1
    
    # 5. EMA Slope (Rate of change of the trend)
    ema_slope = row["EMA20"] - prev_row["EMA20"]
    
    return [ema_gap, rsi, atr_ratio, trend, hour, vol_index, ema_slope]

def record_trade(features, outcome):
    """
    Saves the trade features and outcome (1=Win, 0=Loss) to CSV.
    """
    new_data = pd.DataFrame([features + [outcome]], columns=FEATURES + ["target"])
    
    if not os.path.isfile(DATA_FILE):
        new_data.to_csv(DATA_FILE, index=False)
    else:
        new_data.to_csv(DATA_FILE, mode='a', header=False, index=False)

def train_model():
    """
    Trains the Random Forest using a Balanced Class Weight.
    """
    if not os.path.isfile(DATA_FILE):
        print("AI: No data file found. Skipping training.")
        return

    df = pd.read_csv(DATA_FILE)
    if len(df) < MIN_SAMPLES_TO_TRAIN:
        print(f"AI: Not enough data yet ({len(df)}/{MIN_SAMPLES_TO_TRAIN})")
        return

    X = df[FEATURES]
    y = df["target"]

    # Use Stratified split to keep Win/Loss ratio consistent in training and testing
    test_size = max(0.2, MIN_TEST_SIZE / len(df))
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Random Forest Configuration
    model = RandomForestClassifier(
        n_estimators=300,        # More trees for more features
        max_depth=6,             # Slightly deeper to capture 'Hour' and 'Vol' interactions
        class_weight='balanced', # Crucial: pay extra attention to the fewer "Losses"
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"AI: Training Complete. Samples: {len(df)}. Test Accuracy: {acc:.2%}")

    joblib.dump(model, MODEL_FILE)

def ai_allow_trade(features, threshold=0.58):
    """
    Determines if a trade is statistically likely to win.
    Increased threshold to 0.58 for stricter filtering.
    """
    if not os.path.exists(MODEL_FILE):
        return True # Fail-open: allow trades if model isn't ready
    
    model = joblib.load(MODEL_FILE)
    
    # Check if model expects different number of features
    expected_features = model.n_features_in_
    if len(features) != expected_features:
        print(f"AI: Feature mismatch (got {len(features)}, expected {expected_features}). Retraining...")
        train_model()
        return True  # Allow trade while retraining
    
    # Get probability for class 1 (Win)
    probs = model.predict_proba([features])[0]
    win_prob = probs[1] if len(probs) > 1 else 0
    
    if win_prob >= threshold:
        print(f"AI: TRADE APPROVED (Confidence: {win_prob:.2%})")
        return True
    else:
        print(f"AI: TRADE BLOCKED (Confidence: {win_prob:.2%})")
        return False