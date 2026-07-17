#
# Python Module with Class
# for Vectorized Backtesting
# of Machine Learning-Based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import numpy as np
import pandas as pd
from sklearn import linear_model

import tpqoa
from scipy.optimize import brute

class ScikitVectorBacktester:
    ''' Class for the vectorized backtesting of
    machine learning-based trading strategies.

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
    model: str
        either 'regression' or 'logistic'

    (added)
    granularity: str
        granularity of target data

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_features:
        prepares the features data for the model fitting
    fit_model:
        implements the fitting step
    run_strategy:
        runs the backtest for the regression-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''
    
    def __init__(self, symbol, start, end, amount, tc=0.000025, model='logistic', granularity='D'):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc # for USD_JPN: 0.8Pips(0.008)/2 / 160 = 0.000025

        self.granularity = granularity # example: 'D"

        self.results = None
        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C=1e6,
                solver='lbfgs', max_iter=1000)
        else:
            raise ValueError('Model not known or not yet implemented.')
        
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
            # 特に日足はNYのclose(UTC21:00/22:00)のタイミングでは区切られてはいるが、ラベルとして開始時刻の日付が付けられているために、通常の日中での日付の１日前となっていることに注意。(例：NYベースの2025年の1年間の日足を取るためには、start=2024-12-31とし、end=2025-12-30 を指定する。）
            # ちなみに、この日足と同じ範囲の時間足を取るには（日足のラベルが開始時刻の日付であることを意識すると）Startが前年の12/31の日足ラベルの最初の時間足である2024-12-31 21:00となり、Endは当年の12/30の日足ラベルの最後の時間足である2025-12-31 20:00を指定すればよい。）
            # 但し、時間足を取得する場合に日付を指定することもできるが、この場合にはUTC00:00の時刻指定とみなされることに注意。（例えば、日足を取る場合と同じstart/end日付を指定してしまうと、Start=2025-12-31(00:00)→これは12/31の00:00-20:00の時間足を余分に取り込んでいるだけ、End=2026-12-30(00:00)→これは12/30の1:00の時間足から12/31の20:00の時間足までが取り込まれていない。）
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

        # --- ⑥ 期間でフィルタリング ---
            # 時間足対応は混乱するためやめた。
            # end_dt = pd.to_datetime(self.end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        raw = raw.loc[self.start:self.end]

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
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for the regression and prediction steps.
        '''
        self.data_subset = self.select_data(start,end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = f'lag_{lag}'
            self.data_subset[col] = self.data_subset['returns'].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)

    def fit_model(self, start, end):
        ''' Implements the fitting step.
        '''
        # --- JupyterLab対策としての初期化 ---
        self.feature_columns = []
        self.data_subset = None

        self.prepare_features(start, end)

        # ★ 前日の特徴量を使う（未来リーク防止）
        X = self.data_subset[self.feature_columns].shift(1)
        y = np.sign(self.data_subset['returns'])

        # NaN を除外
        valid = X.notna().all(axis=1)
        X = X[valid]
        y = y[valid]

        # 学習
        self.model.fit(X, y)
        # self.model.fit(self.data_subset[self.feature_columns],
        #                       np.sign(self.data_subset['returns']))

    def run_strategy(self, start_in, end_in, start_out, end_out, lags=3):
        ''' Backtests the trading strategy.
        '''
        self.lags = lags
        self.fit_model(start_in, end_in)

        self.prepare_features(start_out, end_out)

        # ★ 前日の特徴量で予測
        X = self.data_subset[self.feature_columns].shift(1)

        valid = X.notna().all(axis=1)
        X = X[valid]
        self.data_subset = self.data_subset[valid]

        prediction = self.model.predict(X)
        # prediction = self.model.predict(
        #     self.data_subset[self.feature_columns])
        self.data_subset['prediction'] = prediction
        # ★ 当日のPredictionは当日のレコードに書き込まれているを使用する（shift 不要）
        self.data_subset['strategy'] = (self.data_subset['prediction'] * 
                                        self.data_subset['returns'])
        # determine when a trade takes place
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place
        # self.data_subset['strategy'][trades] -= self.tc
        self.data_subset.loc[trades, 'strategy'] -= self.tc
        self.data_subset['creturns'] = (self.amount * 
                        self.data_subset['returns'].cumsum().apply(np.exp))
        self.data_subset['cstrategy'] = (self.amount * \
                        self.data_subset['strategy'].cumsum().apply(np.exp))
        self.results = self.data_subset
        # absolute performance of the strategy
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
        title = f'{self.symbol} | Lags =  {self.lags}'
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10,6)) 

    # 日付は固定でLagsだけOptimizeしたバージョン
    def update_and_run(self, params):
        # brute は params を array(36) のような 0次元配列で渡すことがある
        try:
            lags = int(params)
        except TypeError:
            # もし配列なら中身を取り出す
            lags = int(params.item())

        # すでにセット済みの固定日付を使う
        start_in  = self.start_in
        end_in    = self.end_in
        start_out = self.start_out
        end_out   = self.end_out

        return -self.run_strategy(start_in, end_in, start_out, end_out, lags)[0]


    def optimize_parameters(self, lags_range,
                            start_in, end_in, start_out, end_out):
        """
        使い方のイメージ：
        scibt.optimize_parameters((3, 50, 1),
                                '2002-5-6', '2005-12-31',
                                '2006-1-1', '2009-12-31')
        """

        # 固定の日付をセット
        self.start_in  = start_in
        self.end_in    = end_in
        self.start_out = start_out
        self.end_out   = end_out

        # lags だけを brute で探索
        opt = brute(self.update_and_run,
                    (lags_range,),
                    finish=None)

        best_lags = int(opt)
        best_perf = -self.update_and_run(opt)

        return best_lags, best_perf





if __name__ == '__main__':
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                    10000, 0.0, 'regression')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                    10000, 0.0, 'logistic')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31'))
    print(scibt.run_strategy('2010-1-1', '2016-12-31',
                             '2017-1-1', '2019-12-31'))
    scibt = ScikitVectorBacktester('.SPX', '2010-1-1', '2019-12-31',
                                    10000, 0.001, 'logistic')
    print(scibt.run_strategy('2010-1-1', '2019-12-31',
                             '2010-1-1', '2019-12-31', lags=15))
    print(scibt.run_strategy('2010-1-1', '2013-12-31',
                             '2014-1-1', '2019-12-31', lags=15))