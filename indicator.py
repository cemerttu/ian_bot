# ============================================================
# File: indicator.py
# EMA20 / EMA50 + RSI + ATR + Swing High/Low
# Binary Options – Current Candle Entry
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# SIGNAL ENGINE
# ============================================================
def get_ema_signal(df):
    """
    Returns:
        "BUY", "SELL", or None
    Signal logic uses LAST CLOSED candle only (NO repaint)
    """
    if df is None or len(df) < 60:
        return None

    df = df.copy()

    # ================= INDICATORS =================
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_MA"] = df["ATR"].rolling(14).mean()

    # Swing levels
    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()

    # ================= LAST CLOSED CANDLE =================
    i = -2  # Last closed candle

    close = df.iloc[i]["close"]
    ema20 = df.iloc[i]["EMA20"]
    ema50 = df.iloc[i]["EMA50"]
    rsi = df.iloc[i]["RSI"]
    atr = df.iloc[i]["ATR"]
    atr_ma = df.iloc[i]["ATR_MA"]

    # ================= VOLATILITY FILTER =================
    if atr < atr_ma * 0.9:
        return None

    # ================= REVERSAL =================
    prev_ema20 = df.iloc[i - 1]["EMA20"]
    prev_ema50 = df.iloc[i - 1]["EMA50"]
    prev_rsi = df.iloc[i - 1]["RSI"]

    if prev_ema20 < prev_ema50 and ema20 > ema50 and prev_rsi < 30 and rsi > 35:
        return "BUY"

    if prev_ema20 > prev_ema50 and ema20 < ema50 and prev_rsi > 70 and rsi < 65:
        return "SELL"

    # ================= PULLBACK =================
    if ema20 > ema50 and close <= ema20 and 40 <= rsi <= 55:
        return "BUY"

    if ema20 < ema50 and close >= ema20 and 45 <= rsi <= 60:
        return "SELL"

    # ================= BREAKOUT =================
    if (ema20 > ema50 and close > df.iloc[i - 1]["SwingHigh"] 
        and rsi > 55 and atr > atr_ma * 1.2):
        return "BUY"

    if (ema20 < ema50 and close < df.iloc[i - 1]["SwingLow"] 
        and rsi < 45 and atr > atr_ma * 1.2):
        return "SELL"

    return None

# ============================================================
# BACKTEST ENGINE (BINARY OPTIONS – CURRENT CANDLE)
# ============================================================
def backtest_csv(csv_file, stake=10, payout=0.8):
    """
    Binary Options Backtest:
    - Signal from LAST CLOSED candle
    - Entry at CURRENT candle OPEN
    - Expiry at CURRENT candle CLOSE
    """
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"])

    trades = []

    for i in range(60, len(df) - 1):
        signal = get_ema_signal(df.iloc[: i + 1])

        if signal:
            entry = df.iloc[i + 1]["open"]      # CURRENT candle OPEN
            exit_price = df.iloc[i + 1]["close"] # CURRENT candle CLOSE

            win = exit_price > entry if signal == "BUY" else exit_price < entry
            profit = stake * payout if win else -stake

            trades.append({
                "time": df.iloc[i + 1]["time"],
                "signal": signal,
                "entry": entry,
                "exit": exit_price,
                "profit": profit
            })

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        total_profit = trades_df["profit"].sum()
        win_rate = (trades_df["profit"] > 0).mean() * 100

        print(
            f"Binary Backtest | "
            f"Profit: {total_profit:.2f} | "
            f"Trades: {len(trades_df)} | "
            f"Win Rate: {win_rate:.2f}%"
        )

    return trades_df

# ============================================================
# RUN BACKTEST
# ============================================================
if __name__ == "__main__":
    backtest_csv("EURUSD_M1.csv")
