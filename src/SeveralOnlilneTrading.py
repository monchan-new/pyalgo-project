#
# Online Trading with BacktestLongShort as Order Engine
#
# (c) Shoji Inada
#
import zmq
import datetime
import numpy as np
import pandas as pd
from BacktestLongShort import BacktestLongShort

# --- ZMQ Setup ---
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://0.0.0.0:5555')
socket.setsockopt_string(zmq.SUBSCRIBE, 'SYMBOL')


# --- DataFrame for ticks ---
df = pd.DataFrame()

# --- Strategy parameters ---
mom = 3
SMA1 = 3
SMA2 = 5
MR_SMA = 5
MR_thr = 0.5

# --- Minimum bars needed ---
min_mom = mom + 2
min_sma = SMA2 + 2
min_mr = MR_SMA + 2

# --- Order engine ---
ls = BacktestLongShort('ONLINE', None, None, 10000, verbose=True)

while True:
    # --- Receive tick ---
    data = socket.recv_string()
    t = datetime.datetime.now()
    sym, value = data.split()
    price = float(value)

    # --- Append tick ---
    df = pd.concat([df, pd.DataFrame({sym: price}, index=[t])])

    # --- Build 5-second bars ---
    dr = df.resample('5s', label='right').last()

    dr['returns'] = np.log(dr / dr.shift(1))

    # --- Latest confirmed bar index ---
    last = -3   # 確定終値

    # --- 確定バーを BacktestLongShort に追加（新しいバーが確定した時だけ） ---
    min_all = min(min_mom, min_sma, min_mr)
    if len(dr) > min_all:
        confirmed_time = dr.index[last]
        confirmed_price = dr[sym].iloc[last]
        ls.data.loc[confirmed_time, 'price'] = confirmed_price
        bar = -1 # BacktestLongShort 内の最新行

    # ============================================================
    # 1) MOMENTUM STRATEGY
    # ============================================================
    dr['momentum'] = np.sign(dr['returns'].rolling(mom).mean())

    if len(dr) > min_mom:
        min_mom += 1
        mom_sig = dr['momentum'].iloc[last]
        print("\n=== MOMENTUM SIGNAL ===")
        print(dr.iloc[:-2].tail()) 
        
        if mom_sig == 1.0: 
            print("→ LONG (momentum)")
            ls.go_long(bar, amount='all')
        elif mom_sig == -1.0:
            print("→ SHORT (momentum)")
            ls.go_short(bar, amount='all')

        
    # ============================================================
    # 2) SMA CROSSOVER STRATEGY
    # ============================================================
    dr['SMA1'] = dr[sym].rolling(SMA1).mean()
    dr['SMA2'] = dr[sym].rolling(SMA2).mean()

    if len(dr) > min_sma:
        min_sma += 1
        print("\n=== SMA SIGNAL ===")
        print(dr.iloc[:-2].tail()) 

        if dr['SMA1'].iloc[last] > dr['SMA2'].iloc[last]:
            print("→ LONG (SMA)")
            ls.go_long(bar, amount='all')
        elif dr['SMA1'].iloc[last] < dr['SMA2'].iloc[last]:
            print("→ SHORT (SMA)")
            ls.go_short(bar, amount='all')

    # ============================================================
    # 3) MEAN REVERSION STRATEGY
    # ============================================================
    dr['MR_SMA'] = dr[sym].rolling(MR_SMA).mean()

    if len(dr) > min_mr:
        min_mr += 1
        p = dr[sym].iloc[last]
        s = dr['MR_SMA'].iloc[last]

        print("\n=== MEAN REVERSION SIGNAL ===")
        print(dr.iloc[:-2].tail()) 

        if p < s - MR_thr:
            print("→ LONG (MR)")
            ls.go_long(bar, amount='all')
        elif p > s + MR_thr:
            print("→ SHORT (MR)")
            ls.go_short(bar, amount='all')