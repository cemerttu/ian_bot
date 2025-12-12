# ============================================================
# File: live_chart_mt5_ema_bo_full_v2.py
# EMA20/50 Crossover + Binary Option Signals + Live Chart
# One-row-per-candle logging (open,high,low,close,tick_volume,real_volume,spread)
# ============================================================

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import os

# ===================== CONFIG =======================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1

STAKE = 10
PAYOUT = 0.8
EXPIRY_MINUTES = 1

LIVE_DATA_LOG = "live_data.csv"
TRADE_LOG = "bo_trades.csv"

PLOT_CANDLES = 80
LOOP_SLEEP = 0.25
CANDLE_FETCH = 200

# ===================== MT5 INIT ======================
if not mt5.initialize():
    raise SystemExit("MT5 initialize() failed. Make sure MT5 is running and logged in.")
acct = mt5.account_info()
print("Logged in:", acct.login)

# ===================== GLOBALS & FILE INIT ===========
trades_df = pd.DataFrame(columns=['time','signal','entry','close','profit','expiry'])
trades_lock = threading.Lock()

# create CSVs with full headers if missing
if not os.path.exists(TRADE_LOG):
    pd.DataFrame(columns=['time','signal','entry','close','profit','expiry']).to_csv(TRADE_LOG, index=False)

if not os.path.exists(LIVE_DATA_LOG):
    pd.DataFrame(columns=['time','open','high','low','close','tick_volume','real_volume','spread']).to_csv(LIVE_DATA_LOG, index=False)

# we will track the last saved candle timestamp (epoch int)
_last_saved_candle_ts = None
# track last candle timestamp we opened a trade for (to avoid multiple trades per candle)
_last_traded_candle_ts = None

# ===================== UTILITIES =====================
def save_live_data_latest(df):
    """
    Append the last (most recent) candle to LIVE_DATA_LOG only once per candle.
    Adds: time, open, high, low, close, tick_volume, real_volume, spread
    """
    global _last_saved_candle_ts

    latest = df.tail(1).copy()
    if latest.empty:
        return

    # mt5 rates use integer epoch seconds in 'time'
    candle_ts = int(latest['time'].values[0])

    if candle_ts == _last_saved_candle_ts:
        return  # already saved this candle

    # try to get spread from tick (ask-bid)
    tick = mt5.symbol_info_tick(SYMBOL)
    spread = None
    if tick is not None and hasattr(tick, 'ask') and hasattr(tick, 'bid'):
        try:
            spread = tick.ask - tick.bid
        except Exception:
            spread = None

    # ensure real_volume exists when present; fallback to NaN otherwise
    real_vol = latest['real_volume'].values[0] if 'real_volume' in latest.columns else np.nan
    tick_vol = latest['tick_volume'].values[0] if 'tick_volume' in latest.columns else np.nan

    # format row
    row = {
        'time': pd.to_datetime(candle_ts, unit='s'),
        'open': latest['open'].values[0],
        'high': latest['high'].values[0],
        'low': latest['low'].values[0],
        'close': latest['close'].values[0],
        'tick_volume': tick_vol,
        'real_volume': real_vol,
        'spread': spread
    }

    pd.DataFrame([row]).to_csv(LIVE_DATA_LOG, mode='a', header=False, index=False)
    _last_saved_candle_ts = candle_ts

# ===================== SIGNAL ENGINE =================
def get_signals(df):
    """
    Use EMA20/EMA50, Pullback, Breakout, Reversal logic on the dataframe.
    Returns 'CALL' or 'PUT' or None based on the most recent *completed* candle.
    """
    if df is None or len(df) < 6:
        return None

    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    df["Trend"] = np.where(df['EMA20'] > df['EMA50'], "UP",
                   np.where(df['EMA20'] < df['EMA50'], "DOWN", "NEUTRAL"))

    df["Pullback"] = np.where(
        (df["Trend"]=="UP") & (df["close"] < df["EMA20"]), "CALL",
        np.where((df["Trend"]=="DOWN") & (df["close"] > df["EMA20"]), "PUT", None)
    )

    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()
    df["Breakout"] = np.where(
        df["close"] > df["SwingHigh"].shift(1), "CALL",
        np.where(df["close"] < df["SwingLow"].shift(1), "PUT", None)
    )

    df["Reversal"] = np.where(
        (df["EMA20"].shift(1) < df["EMA50"].shift(1)) & (df["EMA20"] > df["EMA50"]), "CALL",
        np.where((df["EMA20"].shift(1) > df["EMA50"].shift(1)) & (df["EMA20"] < df["EMA50"]), "PUT", None)
    )

    # Check the last **closed** candle (index -2). The last row (-1) is the forming candle.
    # Use the closed candle to avoid signals on incomplete candle/tick noise.
    if len(df) < 2:
        return None

    i = -2
    signals = []
    for col in ["Reversal", "Pullback", "Breakout"]:
        val = df.iloc[i][col]
        if pd.notna(val):
            signals.append(val)

    return signals[-1] if signals else None

# ===================== TRADE HANDLER =================
def place_bo_trade(signal, entry_price, candle_ts):
    """
    Register provisional trade and schedule finish_trade after expiry (non-blocking).
    candle_ts: epoch seconds of the candle we used to trigger the trade (prevents duplicates).
    """
    global _last_traded_candle_ts

    # avoid duplicate trade for same candle
    if candle_ts == _last_traded_candle_ts:
        return False

    expiry_time = datetime.now() + timedelta(minutes=EXPIRY_MINUTES)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE OPENED: {signal} | ENTRY: {entry_price:.5f} | EXPIRY: {expiry_time.strftime('%H:%M:%S')}")

    provisional = {
        'time': datetime.now(),
        'signal': signal,
        'entry': entry_price,
        'close': None,
        'profit': None,
        'expiry': expiry_time
    }

    with trades_lock:
        trades_df.loc[len(trades_df)] = provisional

    # schedule finish
    t = threading.Timer(EXPIRY_MINUTES * 60, finish_trade, args=(signal, entry_price, expiry_time))
    t.daemon = True
    t.start()

    _last_traded_candle_ts = candle_ts
    return True

def finish_trade(signal, entry_price, expiry_time):
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        close_price = tick.bid if signal == "CALL" else tick.ask

        win = (close_price > entry_price) if signal == "CALL" else (close_price < entry_price)
        profit = STAKE * PAYOUT if win else -STAKE

        with trades_lock:
            mask = (trades_df['entry'] == entry_price) & (trades_df['close'].isna())
            if mask.any():
                idx = trades_df[mask].index[0]
                trades_df.at[idx, 'close'] = close_price
                trades_df.at[idx, 'profit'] = profit
                trades_df.at[idx, 'expiry'] = expiry_time
            else:
                # fallback
                trades_df.loc[len(trades_df)] = [datetime.now(), signal, entry_price, close_price, profit, expiry_time]

        # append final row to csv
        pd.DataFrame([{
            'time': datetime.now(),
            'signal': signal,
            'entry': entry_price,
            'close': close_price,
            'profit': profit,
            'expiry': expiry_time
        }]).to_csv(TRADE_LOG, mode='a', header=False, index=False)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE RESULT: {'WIN' if win else 'LOSS'} | Profit: {profit:.2f} | Close: {close_price:.5f}")

    except Exception as e:
        print("Error in finish_trade:", e)

# ===================== PLOT ==========================
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))

def plot_candles(df_all, trades_snapshot):
    ax.clear()
    df_plot = df_all.tail(PLOT_CANDLES).copy()
    if df_plot.empty:
        return

    df_plot['time_dt'] = pd.to_datetime(df_plot['time'], unit='s')
    df_plot['time_num'] = mdates.date2num(df_plot['time_dt'])

    ohlc = df_plot[['time_num', 'open', 'high', 'low', 'close']].values
    time_diffs = df_plot['time_num'].diff().dropna()
    width = (time_diffs.median() if not time_diffs.empty else 0.0005) * 0.6

    candlestick_ohlc(ax, ohlc, width=width, colorup='green', colordown='red', alpha=0.9)

    # EMAs
    df_plot['EMA20'] = df_plot['close'].ewm(span=20, adjust=False).mean()
    df_plot['EMA50'] = df_plot['close'].ewm(span=50, adjust=False).mean()
    ax.plot(df_plot['time_num'], df_plot['EMA20'], color='blue', linewidth=1, label='EMA20')
    ax.plot(df_plot['time_num'], df_plot['EMA50'], color='red', linewidth=1, label='EMA50')

    # Plot trades (arrows under/above candle)
    if not trades_snapshot.empty:
        for _, row in trades_snapshot.iterrows():
            try:
                t_num = mdates.date2num(row['time'])
                if row['signal'] == "CALL":
                    # place slightly below candle low if possible
                    # find matching candle low by nearest time in df_plot
                    nearest = df_plot.iloc[(df_plot['time_dt'] - pd.to_datetime(row['time'])).abs().argsort()[:1]]
                    low = nearest['low'].values[0] if not nearest.empty else row['entry']
                    marker_y = low - (low * 0.00015)
                    ax.scatter(t_num, marker_y, marker='^', s=120, color='#3A7DFF', zorder=6)
                else:
                    nearest = df_plot.iloc[(df_plot['time_dt'] - pd.to_datetime(row['time'])).abs().argsort()[:1]]
                    high = nearest['high'].values[0] if not nearest.empty else row['entry']
                    marker_y = high + (high * 0.00015)
                    ax.scatter(t_num, marker_y, marker='v', s=120, color='#FF8C00', zorder=6)
            except Exception:
                continue

    ax.set_title(f"{SYMBOL} Live Candlestick Chart")
    ax.set_ylabel("Price")
    ax.xaxis_date()
    ax.legend(loc='upper left')
    fig.autofmt_xdate()
    fig.canvas.draw()
    fig.canvas.flush_events()

# ===================== MAIN LOOP =====================
print("LIVE CANDLESTICK BOT STARTED (one-candle-per-row logging)")

try:
    while True:
        # fetch recent candles (most recent includes forming candle)
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLE_FETCH)
        if rates is None or len(rates) == 0:
            time.sleep(1)
            continue

        df = pd.DataFrame(rates)

        # save the latest completed candle only once
        try:
            save_live_data_latest(df)
        except Exception as e:
            print("save_live_data_latest error:", e)

        # Use signals based on last CLOSED candle (avoid forming/tick noise)
        signal = get_signals(df)

        # get the closed candle timestamp used for signal (the closed candle = index -2)
        try:
            closed_candle_ts = int(df.iloc[-2]['time'])
        except Exception:
            closed_candle_ts = None

        if signal and closed_candle_ts is not None:
            tick = mt5.symbol_info_tick(SYMBOL)
            entry_price = tick.ask if signal == "CALL" else tick.bid
            # place trade only once for that closed candle
            placed = place_bo_trade(signal, entry_price, closed_candle_ts)
            # placed == True means we just registered a trade; otherwise ignored as duplicate

        # snapshot trades for plotting
        with trades_lock:
            trades_plot = trades_df.copy()

        # draw chart
        try:
            plot_candles(df, trades_plot)
        except Exception as e:
            print("Plotting error:", e)

        time.sleep(LOOP_SLEEP)

except KeyboardInterrupt:
    print("Stopped by user. Exiting...")

finally:
    mt5.shutdown()
