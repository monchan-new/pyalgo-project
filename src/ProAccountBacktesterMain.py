from BacktestPro import backtest_pro
import tpqoa
import pandas as pd
import numpy as np

# M1データから任意の時間軸（H1, H4, Dなど）のOHLCを生成する汎用関数。
def resample_ohlc_from_m1(df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_m1 : pd.DataFrame
        index が datetime の M1 データ（open, high, low, close が必要）
    timeframe : str
        'H1', 'H4', 'D' など pandas の resample に準拠した時間軸
    
    Returns
    -------
    df_tf : pd.DataFrame
        指定時間軸の OHLC データ
    """

    if timeframe == '4h':
        # OANDA の H4 は 22:00 始まりなので 2時間ずらす
        df_shifted = df_m1.shift(freq='2h')
        df_tf = df_shifted.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close': 'last'
        })
        # 元の時間に戻す
        df_tf.index = df_tf.index - pd.Timedelta(hours=2)
    else:
        df_tf = df_m1.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close': 'last'
        })

    # 欠損行（データが無い時間帯）は削除
    df_tf.dropna(inplace=True)

    return df_tf

# granularity_sma の値を Pandas 形式に変換する辞書
GRANULARITY_MAP = {
    'D': 'D',
    'H4': '4h',
    'H1': '1h',
    'M30': '30min',
    'M15': '15min'
}

# M1データから指定されたGranularityのSMAデータを生成する。
def generate_sma_base_from_m1(df_m1: pd.DataFrame, granularity_sma: str,
                              sma_short: int, sma_long: int) -> pd.DataFrame:
    
    timeframe = GRANULARITY_MAP[granularity_sma]
    
    # --- M1 → 例：日足に変換 ---
    df_tf = resample_ohlc_from_m1(df_m1, timeframe)

    # --- SMA を計算 ---
    df_tf['sma_short'] = df_tf['close'].rolling(sma_short).mean()
    df_tf['sma_long']  = df_tf['close'].rolling(sma_long).mean()

    # --- signal は別で作るのでここでは作らない ---
    return df_tf


# ====== データ取得関数 ======
def fetch_data(api, instrument, granularity, start, end,
               granularity_sma, sma_short, sma_long):
    # Trade対象のデータを取得する。
    df = api.get_history(
        instrument=instrument,
        start=start,
        end=end,
        granularity=granularity,
        price='M'
    )
    df.index = pd.to_datetime(df.index)
    df.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)


    # SMA用のGranularity(例：日足)データをM1データをもとに生成し、短期/長期SMAによるSignalを作成し、上記のTrade対象データに追加する。
    # --- M1データ ---
    df_m1 = api.get_history(
        instrument=instrument,
        start=start,
        end=end,
        granularity='M1',
        price='M'
    )
    df_m1.index = pd.to_datetime(df_m1.index)
    df_m1.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)

    # M1 から SMA 用のGranularity(例：日足)のデータを生成
    df_sma = generate_sma_base_from_m1(df_m1, granularity_sma, sma_short, sma_long)
    # granularity_sma = 'D'
    # df_sma = api.get_history(instrument, start, end, granularity_sma, price='M')
    # df_sma.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)
    # df_sma['sma_short'] = df_sma['close'].rolling(9).mean()
    # df_sma['sma_long']  = df_sma['close'].rolling(26).mean()
    
    
    # SMAを使った signal を生成し、Trade対象のデータにマージする
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
        # SMA側で取得したOANDAのH4の区切りが22:00スタートなので、Tradeデータ側の時間をいったん２時間マイナスした状態で24:00スタートの4h区切りに変換して、最後にその区切りに2時間をプラスする。
        shifted = df.index - pd.Timedelta(hours=2)
        floored = shifted.floor('4h')
        df['sma_time'] = floored + pd.Timedelta(hours=2)
        # df['sma_time'] = df.index.floor('4h')
        df_sma['sma_time'] = df_sma.index
    elif granularity_sma == 'M30':
        df['sma_time'] = df.index.floor('30min')
        df_sma['sma_time'] = df_sma.index
    elif granularity_sma == 'M15':
        df['sma_time'] = df.index.floor('15min')
        df_sma['sma_time'] = df_sma.index
    else:
        raise ValueError("Unsupported granularity")


    # --- SMAのGranularityでマージする ---
    df_sma = df_sma.set_index('sma_time')
    df = df.merge(
        df_sma[['sma_short', 'sma_long', 'signal']], # Positon建ての際にFiltering処理を行うためsma_short/longもmergeするように変更
        left_on='sma_time',
        right_index=True,
        how='left'
)
    # df = df.merge(df_sma[['signal']], left_on='sma_time', 
    #     right_index=True, how='left')

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
    tp_list=[10,20,30],
    # start='2019-11-20',
    start='2026-04-01',
    end='2026-06-01',
    granularity_sma = 'H1'
):

# def run_all_backtests(
#     instruments=['USD_JPY'],
#     granularities=['M5'],
#     sma_settings=[(9,26)],
#     tp_list=[60],
#     start='2019-12-01',
#     end='2026-07-01'
# ):
    
    api = tpqoa.tpqoa('src/pyalgo_netting.cfg')

    results = []

    for inst in instruments:
        for gran in granularities:

            
            for short_sma, long_sma in sma_settings:
                # データ取得
                df = fetch_data(api, inst, gran, start, end, granularity_sma, short_sma, long_sma)

                if df.empty:
                    print(f"{inst} {gran} はデータなし")
                    continue

                # 通貨ペアごとの pip サイズ
                pip = 0.01 if 'JPY' in inst else 0.0001

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
                        'final_pnl': final_pnl,
                        'granularity_sma': granularity_sma
                    })

    return pd.DataFrame(results)


df_results = run_all_backtests()
print(df_results)
