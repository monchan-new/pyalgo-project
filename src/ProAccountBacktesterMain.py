from BacktestPro import backtest_pro
import tpqoa
import pandas as pd
import numpy as np

# ====== データ取得関数 ======
def fetch_data(api, instrument, granularity, start, end):
    df = api.get_history(
        instrument=instrument,
        start=start,
        end=end,
        granularity=granularity,
        price='M'
    )
    df.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)

    # 日足データを取得し短期/長期SMAによるSignalを作成し上記データに追加する。
    df_daily = api.get_history(instrument, start, end, granularity='D', price='M')
    # df_daily = api.get_history(instrument, start, end, granularity='H1', price='M')
    df_daily.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)

    

    df_daily['sma_short'] = df_daily['close'].rolling(9).mean()
    df_daily['sma_long']  = df_daily['close'].rolling(26).mean()
    
    df_daily['signal'] = 0  # 初期値は全部 0（ノーポジ）
    
    mask_valid = df_daily['sma_short'].notna() & df_daily['sma_long'].notna()
    df_daily.loc[mask_valid & (df_daily['sma_short'] > df_daily['sma_long']), 'signal'] = 1
    df_daily.loc[mask_valid & (df_daily['sma_short'] < df_daily['sma_long']), 'signal'] = -1

    # ★ 最初の signal を 0 にする
    df_daily['signal'] = df_daily['signal'].fillna(0)

    # print(df_daily)
    # df_daily.to_csv("data/df_daily.csv")
    
    df = df.merge(df_daily[['signal']], left_index=True, right_index=True, how='left')

    # ① 日足があるバー（22:00）にだけ signal が入る
    # ② それ以外の時間は NaN なので、前の値で埋める
    df['signal'] = df['signal'].ffill()

    # print(df)
    # df.to_csv("data/df.csv")

    return df


# ====== 比較実行関数 ======
def run_all_backtests(
    # instruments=['USD_JPY','EUR_USD','GBP_JPY'],
    instruments=['USD_JPY'],
    # granularities=['M5','H1','H4'],
    granularities=['M5','H1'],
    # sma_settings=[(5,20),(9,26)],
    sma_settings=[(9,26)],
    tp_list=[40,50,60,70],
    start='2026-01-01',
    end='2026-07-01'
):

# def run_all_backtests(
#     instruments=['USD_JPY'],
#     granularities=['H1'],
#     sma_settings=[(9,26)],
#     tp_list=[0],
#     start='2026-01-01',
#     end='2026-07-01'
# ):
    
    api = tpqoa.tpqoa('src/pyalgo.cfg')

    results = []

    for inst in instruments:
        for gran in granularities:

            # データ取得
            df = fetch_data(api, inst, gran, start, end)
            if df.empty:
                print(f"{inst} {gran} はデータなし")
                continue

            # 通貨ペアごとの pip サイズ
            pip = 0.01 if 'JPY' in inst else 0.0001

            for short_sma, long_sma in sma_settings:
                for tp in tp_list:

                    trades, monthly = backtest_pro(
                        df,
                        short_sma=short_sma,
                        long_sma=long_sma,
                        tp_pips=tp,
                        notional=20000,
                        pip=pip
                    )

                    if trades.empty or monthly is None or monthly.empty:
                      print(f"トレードなし: {inst} {gran} SMA({short_sma},{long_sma}) TP={tp}")
                      continue

                  

                    final_pnl = monthly['cum_pnl'].iloc[-1]

                    results.append({
                        'instrument': inst,
                        'granularity': gran,
                        'short_sma': short_sma,
                        'long_sma': long_sma,
                        'tp_pips': tp,
                        'final_pnl': final_pnl
                    })

    return pd.DataFrame(results)


df_results = run_all_backtests()
print(df_results)

