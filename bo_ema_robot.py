import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

# ===================== CONFIG =======================
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
STAKE = 10
PAYOUT = 0.8
EXPIRY_MINUTES = 1

# ===================== MT5 INIT ======================
mt5.initialize()
print("Logged in:", mt5.account_info().login)

# ===================== TRADE DATA ===================
trades_df = pd.DataFrame(columns=['time','signal','entry','close','profit','expiry'])

# ===================== SIGNAL ENGINE =================
def get_signals(df):
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df["Trend"] = np.where(df['EMA20'] > df['EMA50'], "UP",
                   np.where(df['EMA20'] < df['EMA50'], "DOWN", "NEUTRAL"))
    df["Pullback"] = np.where(
        (df["Trend"] == "UP") & (df["close"] < df["EMA20"]), "CALL",
        np.where((df["Trend"] == "DOWN") & (df["close"] > df["EMA20"]), "PUT", None)
    )
    df["SwingHigh"] = df["high"].rolling(5).max()
    df["SwingLow"] = df["low"].rolling(5).min()
    df["Breakout"] = np.where(
        df["close"] > df["SwingHigh"].shift(1), "CALL",
        np.where(df["close"] < df["SwingLow"].shift(1), "PUT", None)
    )
    df["Reversal"] = np.where(
        (df["EMA20"].shift(1) < df["EMA50"].shift(1)) & (df["EMA20"] > df["EMA50"]),
        "CALL",
        np.where((df["EMA20"].shift(1) > df["EMA50"].shift(1)) & (df["EMA20"] < df["EMA50"]), "PUT", None)
    )
    i = len(df) - 1
    signals = []
    for col in ["Reversal", "Pullback", "Breakout"]:
        if df.loc[i, col] is not None:
            signals.append(df.loc[i, col])
    return signals[-1] if signals else None

# ===================== PLACE TRADE =================
def place_bo_trade(signal, entry_price):
    expiry_time = datetime.now() + timedelta(minutes=EXPIRY_MINUTES)
    print(f"TRADE OPENED: {signal} | ENTRY: {entry_price} | EXPIRY: {expiry_time}")
    while datetime.now() < expiry_time:
        time.sleep(1)
    tick = mt5.symbol_info_tick(SYMBOL)
    close_price = tick.bid if signal=="CALL" else tick.ask
    win = (close_price > entry_price) if signal=="CALL" else (close_price < entry_price)
    profit = STAKE*PAYOUT if win else -STAKE
    trades_df.loc[len(trades_df)] = [datetime.now(), signal, entry_price, close_price, profit, expiry_time]
    print(f"TRADE RESULT: {'WIN' if win else 'LOSS'} | Profit: {profit}")

# ===================== LIVE CHART SETUP =================
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))

def plot_candles(df, trades_df):
    ax.clear()
    df_plot = df.copy()
    df_plot['time_dt'] = pd.to_datetime(df_plot['time'], unit='s')
    df_plot['time_num'] = mdates.date2num(df_plot['time_dt'])
    ohlc = df_plot[['time_num','open','high','low','close']].values
    candlestick_ohlc(ax, ohlc, width=0.0005, colorup='green', colordown='red')
    # Plot EMAs
    df_plot['EMA20'] = df_plot['close'].ewm(span=20).mean()
    df_plot['EMA50'] = df_plot['close'].ewm(span=50).mean()
    ax.plot(df_plot['time_num'], df_plot['EMA20'], color='blue', label='EMA20')
    ax.plot(df_plot['time_num'], df_plot['EMA50'], color='red', label='EMA50')
    # Plot trades
    for idx, row in trades_df.iterrows():
        t_num = mdates.date2num(row['time'])
        color = 'blue' if row['signal']=="CALL" else 'orange'
        marker = '^' if row['signal']=="CALL" else 'v'
        ax.scatter(t_num, row['entry'], color=color, marker=marker, s=100)
    ax.xaxis_date()
    ax.legend()
    fig.autofmt_xdate()
    plt.pause(0.01)

# ===================== MAIN LOOP =================
print("LIVE CANDLESTICK BOT STARTED")

while True:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
    if rates is None:
        time.sleep(1)
        continue
    df = pd.DataFrame(rates)
    signal = get_signals(df)
    if signal:
        tick = mt5.symbol_info_tick(SYMBOL)
        entry_price = tick.ask if signal=="CALL" else tick.bid
        place_bo_trade(signal, entry_price)
    plot_candles(df, trades_df)
    time.sleep(1)
