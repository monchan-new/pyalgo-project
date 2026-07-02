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


    # SMA用のGranularity(例：日足)データを取得し短期/長期SMAによるSignalを作成し、上記データに追加する。
    granularity_sma = 'H1'
    df_sma = api.get_history(instrument, start, end, granularity_sma, price='M')
    df_sma.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)
    df_sma['sma_short'] = df_sma['close'].rolling(9).mean()
    df_sma['sma_long']  = df_sma['close'].rolling(26).mean()
    
    df_sma['signal'] = 0  # 初期値を全部 0（ノーポジ）にセットしておくことで、次の処理でSMAがないレコードに対しては処理をスキップできる。
    mask_valid = df_sma['sma_short'].notna() & df_sma['sma_long'].notna()
    df_sma.loc[mask_valid & (df_sma['sma_short'] > df_sma['sma_long']), 'signal'] = 1
    df_sma.loc[mask_valid & (df_sma['sma_short'] < df_sma['sma_long']), 'signal'] = -1

    # --- SMA側のGranularityに応じた時刻メッシュを作成する ---
    if granularity_sma == 'D':
        df['sma_time'] = df.index.floor('D')
        df_sma['sma_time'] = df_sma.index.floor('D') # ここだけ日付のみフォーマットに合わせる必要がある。
    elif granularity_sma == 'H1':
        df['sma_time'] = df.index.floor('h')
        df_sma['sma_time'] = df_sma.index
    elif granularity_sma == 'H4':
        # SMA側で取得したOANDAのH4の区切りが22:00スタートなので、Data側の時間をいったん２時間マイナスした状態で24:00スタートの4h区切りに変換して、最後にその区切りに2時間をプラスする。
        shifted = df.index - pd.Timedelta(hours=2)
        floored = shifted.floor('4h')
        df['sma_time'] = floored + pd.Timedelta(hours=2)
        # df['sma_time'] = df.index.floor('4h')
        df_sma['sma_time'] = df_sma.index
    elif granularity_sma == 'M15':
        df['sma_time'] = df.index.floor('15min')
        df_sma['sma_time'] = df_sma.index
    else:
        raise ValueError("Unsupported granularity")


    # --- SMAのGranularityでマージする ---
    df_sma = df_sma.set_index('sma_time')
    df = df.merge(df_sma[['signal']], left_on='sma_time', right_index=True, how='left')

    # --- signal を ffill してその日の全バーに適用 ---
    df['signal'] = df['signal'].ffill()

    # print(df_sma)
    df_sma.to_csv("data/df_sma.csv")
    # print(df)
    df.to_csv("data/df.csv")

    return df


# ====== 比較実行関数 ======
def run_all_backtests(
    # instruments=['USD_JPY','EUR_USD','GBP_JPY'],
    instruments=['USD_JPY'],
    granularities=['M1'],
    # sma_settings=[(5,20),(9,26)],
    sma_settings=[(9,26)],
    tp_list=[40],
    start='2019-11-20',
    end='2026-06-01'
):

# def run_all_backtests(
#     instruments=['USD_JPY'],
#     granularities=['M5'],
#     sma_settings=[(9,26)],
#     tp_list=[60],
#     start='2019-12-01',
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
