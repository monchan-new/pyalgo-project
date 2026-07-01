import pandas as pd

def backtest_pro(df, short_sma=9, long_sma=26, tp_pips=20, notional=20000, pip=0.01):
    data = df.copy()

    # # SMA
    # data['sma_short'] = data['close'].rolling(short_sma).mean()
    # data['sma_long']  = data['close'].rolling(long_sma).mean()

    # # シグナル（前バーで判定）
    # data['signal'] = 0
    # data.loc[data['sma_short'] > data['sma_long'], 'signal'] = 1
    # data.loc[data['sma_short'] < data['sma_long'], 'signal'] = -1

    position = 0
    entry_price = None
    tp_price = None
    sl_price = None

    trades = []

    # print(df)

    for i in range(1, len(data)):
        prev = data.iloc[i-1]
        row  = data.iloc[i]
        time = data.index[i]

        # ① ポジションなし → 次バー始値でエントリー
        if position == 0:

            # if prev['signal'] == 0:
            #     pass  # 何もしない

            spread = 0.08      # Bid–Ask 全体の幅（8 pips）
            half_spread = spread / 2  # 4 pips = 0.04円

            if prev['signal'] == 1:  # Long
                position = 1
                entry_price = row['open'] + half_spread      # M → Ask
                tp_price = entry_price + tp_pips * pip 
                sl_price = entry_price - tp_pips * pip

            elif prev['signal'] == -1:  # Short
                position = -1
                entry_price = row['open'] - half_spread      # M → Bid
                tp_price = entry_price - tp_pips * pip
                sl_price = entry_price + tp_pips * pip

        # ② ポジションあり → 今バーの高値/安値で OCO 判定
        else:
            if position == 1:  # Long
                if row['low'] <= sl_price:
                    pnl = -tp_pips * notional * pip
                    trades.append((time, 'LONG', entry_price, sl_price, pnl))
                    position = 0
                
                elif row['high'] >= tp_price:
                    pnl = tp_pips * notional * pip
                    trades.append((time, 'LONG', entry_price, tp_price, pnl))
                    position = 0


            elif position == -1:  # Short
                if row['high'] >= sl_price:
                    pnl = -tp_pips * notional * pip
                    trades.append((time, 'SHORT', entry_price, sl_price, pnl))
                    position = 0
                
                elif row['low'] <= tp_price:
                    pnl = tp_pips * notional * pip
                    trades.append((time, 'SHORT', entry_price, tp_price, pnl))
                    position = 0

    # 結果を DataFrame に
    trades_df = pd.DataFrame(trades, columns=['time_exit', 'side', 'entry', 'exit', 'pnl'])
    trades_df['time_exit'] = pd.to_datetime(trades_df['time_exit'])
    trades_df['month'] = trades_df['time_exit'].dt.to_period('M')

    if trades_df.empty:
        return trades_df, None

    monthly = trades_df.groupby('month')['pnl'].sum().to_frame('monthly_pnl')
    monthly['cum_pnl'] = monthly['monthly_pnl'].cumsum()
    monthly['trade_count'] = trades_df.groupby('month')['pnl'].count()

    
    # print(trades_df, monthly)
    print(monthly)
    # trades_df.to_csv("data/trades_df.csv")
    
    return trades_df, monthly
