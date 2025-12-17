# ============================================================
# File: indicator.py
# EMA20/50 Signal Engine + Historical CSV Backtest
# ============================================================

import pandas as pd
import numpy as np

# ================== SIGNAL ENGINE =========================
def get_ema_signal(df):
    """
    Returns a signal ("BUY"/"SELL"/None) for the last closed candle based on EMA20/EMA50 logic.
    """
    if df is None or len(df) < 6:
        return None

    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    # Trend direction
    df["Trend"] = np.where(df['EMA20'] > df['EMA50'], "UP",
                   np.where(df['EMA20'] < df['EMA50'], "DOWN", "NEUTRAL"))

    # Pullback
    df["Pullback"] = np.where(
        (df["Trend"]=="UP") & (df["close"] < df["EMA20"]), "BUY",
        np.where((df["Trend"]=="DOWN") & (df["close"] > df["EMA20"]), "SELL", None)
    )

    # Swing highs/lows for breakout
    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()
    df["Breakout"] = np.where(
        df["close"] > df["SwingHigh"].shift(1), "BUY",
        np.where(df["close"] < df["SwingLow"].shift(1), "SELL", None)
    )

    # EMA cross reversal
    df["Reversal"] = np.where(
        (df["EMA20"].shift(1) < df["EMA50"].shift(1)) & (df["EMA20"] > df["EMA50"]), "BUY",
        np.where((df["EMA20"].shift(1) > df["EMA50"].shift(1)) & (df["EMA20"] < df["EMA50"]), "SELL", None)
    )

    # Return last closed candle signal
    if len(df) < 2:
        return None

    i = -2  # last closed candle
    signals = []
    for col in ["Reversal", "Pullback", "Breakout"]:
        val = df.iloc[i][col]
        if pd.notna(val):
            signals.append(val)

    return signals[-1] if signals else None


# ================== BACKTEST =============================
def backtest_csv(csv_file, stake=10, payout=0.8):
    """
    Backtest EMA20/50 signals on historical EURUSD 1-minute CSV.
    CSV must have: time, open, high, low, close
    """
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    trades = []

    for i in range(1, len(df)):
        signal = get_ema_signal(df.iloc[:i+1])
        if signal:
            entry_price = df.iloc[i]['close']
            close_price = df.iloc[i+1]['close'] if i+1 < len(df) else df.iloc[i]['close']
            win = (close_price > entry_price) if signal == "BUY" else (close_price < entry_price)
            profit = stake * payout if win else -stake
            trades.append({
                'time': df.iloc[i]['time'],
                'signal': signal,
                'entry': entry_price,
                'close': close_price,
                'profit': profit
            })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        total_profit = trades_df['profit'].sum()
        win_rate = len(trades_df[trades_df['profit']>0]) / len(trades_df) * 100
        print(f"Backtest Result: Total Profit: {total_profit:.2f}, Trades: {len(trades_df)}, Win Rate: {win_rate:.2f}%")
    return trades_df


# Optional quick test
if __name__ == "__main__":
    backtest_csv("EURUSD_M1.csv")
