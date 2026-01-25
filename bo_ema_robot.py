# ============================================================
# EMA20/50 Scalper Binary Option Robot (With News + Session Filter)
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

from indicator import get_ema_signal, add_indicators
from ai_filter import extract_features, ai_allow_trade, record_trade, train_model
from news_filter import news_block_active
from session_filter import session_allowed

# ===================== CONFIG =======================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1

STAKE = 10
PAYOUT = 0.8
EXPIRY_MINUTES = 2

LIVE_DATA_LOG = "live_data.csv"
TRADE_LOG = "bo_trades.csv"

PLOT_CANDLES = 80
LOOP_SLEEP = 0.2
CANDLE_FETCH = 200

# ===================== MT5 INIT ======================
if not mt5.initialize():
    raise SystemExit("MT5 initialize() failed")

print("Logged in:", mt5.account_info().login)

# ===================== GLOBALS =======================
trades_df = pd.DataFrame(columns=['time','signal','entry','close','profit','expiry'])
trades_lock = threading.Lock()

_last_saved_candle_ts = None
_last_traded_candle_ts = None
_trade_in_progress = False

# Create files if missing
for file, cols in [
    (TRADE_LOG, trades_df.columns),
    (LIVE_DATA_LOG, ['time','open','high','low','close','tick_volume','real_volume','spread'])
]:
    if not os.path.exists(file):
        pd.DataFrame(columns=cols).to_csv(file, index=False)

# ===================== UTILITIES =====================
def save_live_data_latest(df):
    global _last_saved_candle_ts
    latest = df.tail(1).copy()
    if latest.empty:
        return

    candle_ts = int(latest['time'].values[0])
    if candle_ts == _last_saved_candle_ts:
        return

    row = {
        'time': pd.to_datetime(candle_ts, unit='s'),
        'open': latest['open'].values[0],
        'high': latest['high'].values[0],
        'low': latest['low'].values[0],
        'close': latest['close'].values[0],
        'tick_volume': latest.get('tick_volume', np.nan),
        'real_volume': latest.get('real_volume', np.nan),
        'spread': np.nan
    }

    pd.DataFrame([row]).to_csv(LIVE_DATA_LOG, mode='a', header=False, index=False)
    _last_saved_candle_ts = candle_ts

# ===================== TRADE HANDLER =================
def place_bo_trade(signal, entry_price, candle_ts, df_snapshot):
    global _last_traded_candle_ts, _trade_in_progress

    if _trade_in_progress or candle_ts == _last_traded_candle_ts:
        return False

    features = extract_features(df_snapshot)
    if not ai_allow_trade(features):
        print("🤖 AI BLOCKED")
        return False

    _trade_in_progress = True
    _last_traded_candle_ts = candle_ts
    expiry_time = datetime.now() + timedelta(minutes=EXPIRY_MINUTES)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] TRADE OPENED: {signal} @ {entry_price:.5f}")

    with trades_lock:
        trades_df.loc[len(trades_df)] = {
            'time': datetime.now(),
            'signal': signal,
            'entry': entry_price,
            'close': None,
            'profit': None,
            'expiry': expiry_time
        }

    t = threading.Timer(
        EXPIRY_MINUTES * 60,
        finish_trade,
        args=(signal, entry_price, expiry_time, features)
    )
    t.daemon = True
    t.start()
    return True

def finish_trade(signal, entry_price, expiry_time, features):
    global _trade_in_progress

    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
        close_price = rates[0]['close']

        win = close_price > entry_price if signal == "BUY" else close_price < entry_price
        profit = STAKE * PAYOUT if win else -STAKE

        with trades_lock:
            mask = (trades_df['entry'] == entry_price) & (trades_df['close'].isna())
            if mask.any():
                idx = trades_df[mask].index[0]
                trades_df.at[idx, 'close'] = close_price
                trades_df.at[idx, 'profit'] = profit

        pd.DataFrame([{
            'time': datetime.now(),
            'signal': signal,
            'entry': entry_price,
            'close': close_price,
            'profit': profit,
            'expiry': expiry_time
        }]).to_csv(TRADE_LOG, mode='a', header=False, index=False)

        record_trade(features, win)
        train_model()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] RESULT: {'WIN' if win else 'LOSS'} | {profit}")

    except Exception as e:
        print("Trade error:", e)
    finally:
        _trade_in_progress = False

# ===================== PLOT =========================
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))

def plot_candles(df_all, trades_snapshot):
    ax.clear()

    df_plot = df_all.tail(PLOT_CANDLES).copy()
    if df_plot.empty:
        return

    df_plot['time_dt'] = pd.to_datetime(df_plot['time'], unit='s')
    df_plot['time_num'] = mdates.date2num(df_plot['time_dt'])

    ohlc = df_plot[['time_num','open','high','low','close']].values
    width = (df_plot['time_num'].diff().median() or 0.0005) * 0.6
    candlestick_ohlc(ax, ohlc, width=width, colorup='green', colordown='red', alpha=0.9)

    df_plot['EMA20'] = df_plot['close'].ewm(span=20, adjust=False).mean()
    df_plot['EMA50'] = df_plot['close'].ewm(span=50, adjust=False).mean()

    ax.plot(df_plot['time_num'], df_plot['EMA20'], color='blue', linewidth=1)
    ax.plot(df_plot['time_num'], df_plot['EMA50'], color='red', linewidth=1)

    for _, row in trades_snapshot.iterrows():
        entry_time = mdates.date2num(row['time'])
        ax.scatter(entry_time, row['entry'],
                   marker='^' if row['signal']=="BUY" else 'v',
                   color='green' if row['signal']=="BUY" else 'red',
                   s=80, zorder=5)

    ax.set_title(f"{SYMBOL} M1 Scalper")
    ax.xaxis_date()
    fig.autofmt_xdate()
    fig.canvas.draw()
    fig.canvas.flush_events()

# ===================== MAIN LOOP =====================
print("SCALPER CANDLESTICK BOT STARTED")
 
try:
    while True:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, CANDLE_FETCH)
        if rates is None or len(rates) == 0:
            time.sleep(1)
            continue

        df = pd.DataFrame(rates)
        df = add_indicators(df)
        save_live_data_latest(df)

        signal = get_ema_signal(df)
        closed_candle_ts = int(df.iloc[-2]['time']) if len(df) > 1 else None

        if signal and closed_candle_ts is not None:

            if not session_allowed():
                print("⏰ SESSION BLOCK")
            elif news_block_active(SYMBOL):
                print("📰 NEWS BLOCK")
            else:
                entry_price = df.iloc[-2]['close']
                place_bo_trade(signal, entry_price, closed_candle_ts, df.copy())

        with trades_lock:
            trades_plot = trades_df.copy()

        plot_candles(df, trades_plot)
        time.sleep(LOOP_SLEEP)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    mt5.shutdown()
    print("MT5 shutdown completed")