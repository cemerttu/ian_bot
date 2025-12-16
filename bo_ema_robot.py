# ============================================================
# File: live_chart_mt5_ema_bo_full_v4.py
# EMA20/50 Crossover + Binary Option Signals + Live Chart
# One-row-per-candle logging (open,high,low,close,tick_volume,real_volume,spread)
# ONE ACTIVE TRADE AT A TIME
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

# ===================== GLOBALS & FILE INIT =============
trades_df = pd.DataFrame(columns=['time','signal','entry','close','profit','expiry'])
trades_lock = threading.Lock()

_last_saved_candle_ts = None
_last_traded_candle_ts = None
_trade_in_progress = False  # 🔴 Ensures only one active trade

# create CSVs with full headers if missing
if not os.path.exists(TRADE_LOG):
    pd.DataFrame(columns=['time','signal','entry','close','profit','expiry']).to_csv(TRADE_LOG, index=False)

if not os.path.exists(LIVE_DATA_LOG):
    pd.DataFrame(columns=['time','open','high','low','close','tick_volume','real_volume','spread']).to_csv(LIVE_DATA_LOG, index=False)

# ===================== UTILITIES =======================
def save_live_data_latest(df):
    global _last_saved_candle_ts
    latest = df.tail(1).copy()
    if latest.empty:
        return
    candle_ts = int(latest['time'].values[0])
    if candle_ts == _last_saved_candle_ts:
        return  # already saved

    tick = mt5.symbol_info_tick(SYMBOL)
    spread = tick.ask - tick.bid if tick else None

    row = {
        'time': pd.to_datetime(candle_ts, unit='s'),
        'open': latest['open'].values[0],
        'high': latest['high'].values[0],
        'low': latest['low'].values[0],
        'close': latest['close'].values[0],
        'tick_volume': latest.get('tick_volume', np.nan),
        'real_volume': latest.get('real_volume', np.nan),
        'spread': spread
    }
    pd.DataFrame([row]).to_csv(LIVE_DATA_LOG, mode='a', header=False, index=False)
    _last_saved_candle_ts = candle_ts

# ===================== SIGNAL ENGINE ====================
def get_signals(df):
    if df is None or len(df) < 6:
        return None

    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    df["Trend"] = np.where(df['EMA20'] > df['EMA50'], "UP",
                   np.where(df['EMA20'] < df['EMA50'], "DOWN", "NEUTRAL"))

    df["Pullback"] = np.where(
        (df["Trend"]=="UP") & (df["close"] < df["EMA20"]), "BUY",
        np.where((df["Trend"]=="DOWN") & (df["close"] > df["EMA20"]), "SELL", None)
    )

    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()
    df["Breakout"] = np.where(
        df["close"] > df["SwingHigh"].shift(1), "BUY",
        np.where(df["close"] < df["SwingLow"].shift(1), "SELL", None)
    )

    df["Reversal"] = np.where(
        (df["EMA20"].shift(1) < df["EMA50"].shift(1)) & (df["EMA20"] > df["EMA50"]), "BUY",
        np.where((df["EMA20"].shift(1) > df["EMA50"].shift(1)) & (df["EMA20"] < df["EMA50"]), "SELL", None)
    )

    if len(df) < 2:
        return None

    i = -2  # last closed candle
    signals = []
    for col in ["Reversal", "Pullback", "Breakout"]:
        val = df.iloc[i][col]
        if pd.notna(val):
            signals.append(val)

    return signals[-1] if signals else None

# ===================== TRADE HANDLER ===================
def place_bo_trade(signal, entry_price, candle_ts):
    global _last_traded_candle_ts, _trade_in_progress

    if _trade_in_progress:
        return False

    if candle_ts == _last_traded_candle_ts:
        return False

    _trade_in_progress = True
    _last_traded_candle_ts = candle_ts

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

    t = threading.Timer(EXPIRY_MINUTES * 60, finish_trade, args=(signal, entry_price, expiry_time))
    t.daemon = True
    t.start()

    return True

def finish_trade(signal, entry_price, expiry_time):
    global _trade_in_progress
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        close_price = tick.bid if signal == "BUY" else tick.ask

        win = (close_price > entry_price) if signal == "BUY" else (close_price < entry_price)
        profit = STAKE * PAYOUT if win else -STAKE

        with trades_lock:
            mask = (trades_df['entry'] == entry_price) & (trades_df['close'].isna())
            if mask.any():
                idx = trades_df[mask].index[0]
                trades_df.at[idx, 'close'] = close_price
                trades_df.at[idx, 'profit'] = profit
                trades_df.at[idx, 'expiry'] = expiry_time
            else:
                trades_df.loc[len(trades_df)] = [datetime.now(), signal, entry_price, close_price, profit, expiry_time]

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

    finally:
        _trade_in_progress = False  # allow next trade

# ===================== PLOT ===========================
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
  
    df_plot['EMA20'] = df_plot['close'].ewm(span=20, adjust=False).mean()
    df_plot['EMA50'] = df_plot['close'].ewm(span=50, adjust=False).mean()
    ax.plot(df_plot['time_num'], df_plot['EMA20'], color='blue', linewidth=1, label='EMA20')
    ax.plot(df_plot['time_num'], df_plot['EMA50'], color='red', linewidth=1, label='EMA50')

    if not trades_snapshot.empty:
        for _, row in trades_snapshot.iterrows():
            try:
                t_num = mdates.date2num(row['time'])
                if row['signal'] == "BUY":
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

# ===================== MAIN LOOP =======================
print("LIVE CANDLESTICK BOT STARTED (one-candle-per-row logging)")

try:
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLE_FETCH)
        if rates is None or len(rates) == 0:
            time.sleep(1)
            continue

        df = pd.DataFrame(rates)

        try:
            save_live_data_latest(df)
        except Exception as e:
            print("save_live_data_latest error:", e)

        signal = get_signals(df)
        try:
            closed_candle_ts = int(df.iloc[-2]['time'])
        except Exception:
            closed_candle_ts = None

        if signal and closed_candle_ts is not None:
            tick = mt5.symbol_info_tick(SYMBOL)
            entry_price = tick.ask if signal == "BUY" else tick.bid
            place_bo_trade(signal, entry_price, closed_candle_ts)

        with trades_lock:
            trades_plot = trades_df.copy()

        try:
            plot_candles(df, trades_plot)
        except Exception as e:
            print("Plotting error:", e)

        time.sleep(LOOP_SLEEP)

except KeyboardInterrupt:
    print("Stopped by user. Exiting...")

finally:
    mt5.shutdown()
 