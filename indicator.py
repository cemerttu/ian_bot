# ============================================================
# File: indicator.py
# EMA20 / EMA50 + RSI + ATR + Swing High/Low
# Binary Options Scalper – Current Candle Entry + Backtest
# ============================================================

import pandas as pd
import numpy as np

# ================== SIGNAL ENGINE =========================
def add_indicators(df):
    df = df.copy()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta>0,0.0)
    loss = -delta.where(delta<0,0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100/(1+rs))

    # ATR
    tr = pd.concat([df["high"]-df["low"], (df["high"]-df["close"].shift()).abs(), (df["low"]-df["close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_MA"] = df["ATR"].rolling(14).mean()

    # Swing levels
    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()
    return df


def get_ema_signal(df):
    if df is None or len(df) < 50:
        return None

    df = add_indicators(df)

    i=-2
    close = df.iloc[i]["close"]
    ema20 = df.iloc[i]["EMA20"]
    ema50 = df.iloc[i]["EMA50"]
    rsi = df.iloc[i]["RSI"]
    atr = df.iloc[i]["ATR"]
    atr_ma = df.iloc[i]["ATR_MA"]

    if atr < atr_ma*0.8:
        return None

    prev_ema20 = df.iloc[i-1]["EMA20"]
    prev_ema50 = df.iloc[i-1]["EMA50"]
    prev_rsi = df.iloc[i-1]["RSI"]

    # REVERSAL
    if prev_ema20 < prev_ema50 and ema20 > ema50 and prev_rsi < 35 and rsi > 40:
        return "BUY"
    if prev_ema20 > prev_ema50 and ema20 < ema50 and prev_rsi > 65 and rsi < 60:
        return "SELL"

    # PULLBACK
    if ema20 > ema50 and close <= ema20 and 40 <= rsi <= 55:
        return "BUY"
    if ema20 < ema50 and close >= ema20 and 45 <= rsi <= 60:
        return "SELL"

    # BREAKOUT
    if ema20 > ema50 and close > df.iloc[i-1]["SwingHigh"] and rsi>55 and atr>atr_ma*1.1:
        return "BUY"
    if ema20 < ema50 and close < df.iloc[i-1]["SwingLow"] and rsi<45 and atr>atr_ma*1.1:
        return "SELL"

    return None

# ================== BACKTEST ENGINE =========================
def backtest_csv(csv_file, stake=10, payout=0.8):
    df = pd.read_csv(csv_file)
    df["time"] = pd.to_datetime(df["time"])
    trades = []

    for i in range(50, len(df)-1):
        signal = get_ema_signal(df.iloc[:i+1])
        if signal:
            entry = df.iloc[i+1]["open"]
            exit_price = df.iloc[i+1]["close"]
            win = exit_price>entry if signal=="BUY" else exit_price<entry
            profit = stake*payout if win else -stake
            trades.append({"time":df.iloc[i+1]["time"],"signal":signal,"entry":entry,"exit":exit_price,"profit":profit})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_profit = trades_df["profit"].sum()
        win_rate = (trades_df["profit"]>0).mean()*100
        print(f"Binary Scalper Backtest | Profit: {total_profit:.2f} | Trades: {len(trades_df)} | Win Rate: {win_rate:.2f}%")
    return trades_df

if __name__=="__main__":
    backtest_csv("EURUSD_M1.csv")
