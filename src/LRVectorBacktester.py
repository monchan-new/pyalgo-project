#
# Python Module with Class
# for Vectorized Backtesting
# of Linear Regression-Based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd

import tpqoa
from scipy.optimize import brute

class LRVectorBacktester:
    ''' Class for the vectorized backtesting of
    linear regression-based trading strategies.

    Attributes
    ==========
    symbol: str
       TR RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade

    (added)
    granularity: str
        granularity of target data

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_lags:
        prepares the lagged data for the regression
    fit_model:
        implements the regression step
    run_strategy:
        runs the backtest for the regression-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''
    
    def __init__(self, symbol, start, end, amount, tc=0.000025, granularity='D'):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc # for USD_JPN: 0.8Pips(0.008)/2 / 160 = 0.000025
        self.granularity = granularity # example: 'D"
        self.results = None

        # API は1回だけ作る
        self.api = tpqoa.tpqoa('/workspace/src/pyalgo_netting.cfg')

        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        # --- ① OANDA からヒストリカルデータ取得 ---
        df = self.api.get_history(
            instrument=self.symbol,     # 例: 'EUR_USD'
            # 日付及び時刻はUTCのベースの日付／時刻を指定する必要がある。
            # 特に日足はNYのclose(UTC21:00/21:00)のタイミングでは区切られてはいるが、開始時刻の日付が付けられているために、通常感覚の日付の１日前の日付となっていることに注意。
            # 時間足に日付を指定した場合にはUTC00:00の時刻指定とみなされるが、end日に対しては(日足と同様にその日全体を含むようにするために)UTC23:59まで延長するようにしている。
            start=self.start, # '2009-12-31', '2009-12-31T21:00:00Z'
            end=self.end,     # '2010-12-30', "2010-12-31T20:59:59Z"
            granularity=self.granularity, # 'D', 'H1', 'M15', 'S5' など
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

        # --- ⑥ 期間でフィルタリング（時間足対応版） ---
        end_dt = pd.to_datetime(self.end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        raw = raw.loc[self.start:end_dt]
        # # --- ⑥ 期間でフィルタリング --- (end日を含むように変更←不要だったみたい)---
        # end_dt = pd.to_datetime(self.end) + pd.Timedelta(days=1)
        # raw = raw.loc[self.start:end_dt]

        # raw = pd.read_csv(
        #     'https://hilpisch.com/pyalgo_eikon_eod_data.csv',
        #     index_col=0, parse_dates=True
        # ).dropna()
        # raw = pd.DataFrame(raw[self.symbol])
        # raw = raw.loc[self.start:self.end]
        # raw.rename(columns={self.symbol: 'price'}, inplace=True)

        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data.
        '''
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data
    
    def prepare_lags(self, start, end):
        ''' Prepares the lagged data for the regression and prediction steps.
        '''
        data = self.select_data(start,end)
        self.cols = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            data[col] = data['returns'].shift(lag)
            self.cols.append(col)
        data.dropna(inplace=True)
        self.lagged_data = data

    def fit_model(self, start, end):
        ''' Implements the regression step.
        '''
        self.prepare_lags(start, end)
        reg = np.linalg.lstsq(self.lagged_data[self.cols],
                              np.sign(self.lagged_data['returns']),
                              rcond=None)[0]
        self.reg = reg

    def print_reg(self):
        print(self.reg)

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
        '''
        self.start_in = start_in
        self.end_in = end_in
        self.start_out = start_out
        self.end_out = end_out

        self.lags = lags
        self.fit_model(start_in, end_in)
        self.results = self.select_data(start_out, end_out).iloc[lags:]
        self.prepare_lags(start_out, end_out)
        prediction = np.sign(np.dot(self.lagged_data[self.cols], self.reg))
        self.results['prediction'] = prediction
        self.results['strategy'] = self.results['prediction'] * \
                                   self.results['returns']
        # determine when a trade takes place
        trades = self.results['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        # self.results['strategy'][trades] -= self.tc
        self.results.loc[trades, 'strategy'] -= self.tc
        self.results['creturns'] = self.amount * \
                        self.results['returns'].cumsum().apply(np.exp)
        self.results['cstrategy'] = self.amount * \
                        self.results['strategy'].cumsum().apply(np.exp)
        # gross performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = f'{self.symbol} | TC =  {self.tc:.6f}'
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10,6)) 
        

    def optimize_parameters(self, lags_range):
        best_lags = None
        best_value = -np.inf

        start, end, step = lags_range

        for m in range(start, end + 1, step):
            value = self.run_strategy(
                self.start_in,
                self.end_in,
                self.start_out,
                self.end_out,
                lags=m
            )[0]
            if value > best_value:
                best_value = value
                best_lags = m

        return best_lags, best_value


if __name__ == '__main__':
    lrbt = LRVectorBacktester('.SPX', '2010-1-1', '2018-06-29', 10000, 0.0)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(lrbt.run_strategy('2010-1-1', '2015-12-31',
                             '2016-1-1', '2019-12-31'))
    lrbt = LRVectorBacktester('GDX', '2010-1-1', '2019-12-31', 10000, 0.001)
    print(lrbt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31', lags=5))
    print(lrbt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31', lags=5))