import numpy as np
import pandas as pd
from scipy.optimize import brute

import tpqoa

class SMAVectorBacktester:

    def __init__(self, symbol, SMA1, SMA2, start, end, granularity='D'):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end

        self.granularity = granularity
        
        self.results = None

        # API は1回だけ作る（重要）
        self.api = tpqoa.tpqoa('/workspace/src/pyalgo_netting.cfg')
        
        self.get_data()

    def get_data(self):
        # --- ① OANDA からヒストリカルデータ取得 ---
        df = self.api.get_history(
            instrument=self.symbol,     # 例: 'EUR_USD'
            start=self.start,           # '2010-01-01'
            end=self.end,               # '2020-01-01'
            granularity=self.granularity, # 'D', 'H1', 'M15' など
            price='M'                   # Mid価格（OHLC）
        )

        # --- ② 未確定足を除外（バックテストでは必須） ---
        df = df[df['complete'] == True]

        # --- ③ Index を datetime に変換 ---
        df.index = pd.to_datetime(df.index)

        # --- ④ 列名を統一（Hilpisch のコードと互換にする） ---
        df.rename(columns={'o':'open','h':'high','l':'low','c':'close'}, inplace=True)

        # --- ⑤ Hilpisch の raw と同じ構造に変換 ---
        raw = pd.DataFrame(df['close'])
        raw.rename(columns={'close': 'price'}, inplace=True)


        # --- ⑥ 期間でフィルタリング (end日を含むように変更をやめた)---
        # end_dt = pd.to_datetime(self.end) + pd.Timedelta(days=1)
        end_dt = pd.to_datetime(self.end)
        raw = raw.loc[self.start:end_dt]

        # raw = pd.read_csv(
        #     'https://hilpisch.com/pyalgo_eikon_eod_data.csv',
        #     index_col=0, parse_dates=True
        # ).dropna()

        # raw = pd.DataFrame(raw[self.symbol])
        # raw = raw.loc[self.start:self.end]
        # raw.rename(columns={self.symbol: 'price'}, inplace=True)

        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()

        self.data = raw
        # print(raw)

    def set_parameters(self, SMA1=None, SMA2=None):
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        # 最初の行はreturnはなしでPrice/SMAのみ存在する行となるが、Signalとして使用可能と判断し、dropna()しないように変更。
        data = self.data.copy()
        # data = self.data.copy().dropna()
        # print('before position',data)

        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        # data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
        data['strategy'] = data['position'].shift(1) * data['return']

        # print('before dropna',data)
        data.dropna(inplace=True)
        # print('after dropna',data)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)

        self.results = data
        # print("data::", data)

        aperf = data['cstrategy'].iloc[-1]
        operf = aperf - data['creturns'].iloc[-1]
        
        print("SMA1 & SMA2:", self.SMA1, self.SMA2)
        print("Absolute performance:", aperf)

        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
            return

        title = f"{self.symbol} | SMA1={self.SMA1}, SMA2={self.SMA2}"
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))

    def update_and_run(self, SMA):
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)
