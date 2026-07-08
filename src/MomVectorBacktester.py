#
# Python Module with Class
# for Vectorized Backtesting
# of Momentum-Based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd

import tpqoa
from scipy.optimize import brute

class MomVectorBacktester:
    ''' Class for the vectorized backtesting of
    momentum-based trading strategies.

    Attributes
    ==========
    symbol: str
       RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    run_strategy:
        runs the backtest for the momentum-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''

    def __init__(self, symbol, start, end, amount, tc=0.000025, granularity='D'):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc # for USD_JPN: 0.8Pips(0.008)/2 / 160 = 0.000025
        self.granularity = granularity
        self.results = None

        # API は1回だけ作る（重要）
        self.api = tpqoa.tpqoa('/workspace/src/pyalgo_netting.cfg')

        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
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

        # --- ⑥ 期間でフィルタリング ---
        raw = raw.loc[self.start:self.end]


        # raw = pd.read_csv(
        #     'https://hilpisch.com/pyalgo_eikon_eod_data.csv',
        #     index_col=0, parse_dates=True
        # ).dropna()

        # raw = pd.DataFrame(raw[self.symbol])
        # raw = raw.loc[self.start:self.end]
        # raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        self.data = raw

    def run_strategy(self, momentum=1):
        ''' Backtests the trading strategy.
        '''
        self.momentum = momentum
        # data = self.data.copy().dropna()
        data = self.data.copy()
        data.dropna(inplace=True)

        data['position'] = np.sign(data['return'].rolling(momentum).mean())
        data['strategy'] = data['position'].shift(1) * data['return']
        # determine when a trade takes place
        data.dropna(inplace=True)
        trades = data['position'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        data.loc[trades,'strategy'] -= self.tc
        data['creturns'] = self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)

        self.results = data
        # absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]


        trades = (data['position'].diff() != 0).sum()
        print("Number of trades:", trades)
        print("Absolute performance:", aperf)

        return round(aperf, 2), round(operf, 2)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
            return

        title = f'{self.symbol} | TC = {self.tc:.4f}'
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))


    
    # def set_parameters(self, momentum=None):
    #     if momentum is not None:
    #         self.momentum = momentum

    # def update_and_run(self, params):
    #     momentum = int(params[0])  # 1つ目だけ使う
    #     self.set_parameters(momentum)
    #     return -self.run_strategy()[0]

    def optimize_parameters(self, momentum_range):
        best_momentum = None
        best_value = -np.inf

        start, end, step = momentum_range

        for m in range(start, end + 1, step):
            value = self.run_strategy(momentum=m)[0]
            if value > best_value:
                best_value = value
                best_momentum = m

        return best_momentum, best_value


    

if __name__ == '__main__':
    mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31',
                                10000, 0.0)
    print(mombt.run_strategy())
    print(mombt.run_strategy(momentum=2))
    mombt = MomVectorBacktester('XAU=', '2010-1-1', '2020-12-31',
                                10000, 0.001)
    print(mombt.run_strategy(momentum=2))