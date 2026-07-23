#
# Python Module with Class
# for Vectorized Backtesting
# of Mean-Reversion Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
from MomVectorBacktester import *


class MRVectorBacktester(MomVectorBacktester):
    ''' Class for the vectorized backtesting of
    mean reversion-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC symbol with which to work
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g., 0.5% = 0.005) per trade

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    run_strategy:
        runs the backtest for the mean reversion-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''

    def run_strategy(self, SMA, threshold):
        ''' Backtests the trading strategy.
        '''
        # 最初の行はreturnはなしでPriceのみ存在する行となるが、Signalとして使用可能と判断し、dropna()しないように変更。
        data = self.data.copy()
        # data = self.data.copy().dropna()
        data['sma'] = data['price'].rolling(SMA).mean()
        data['distance'] = data['price'] - data['sma']
        # 標準偏差の倍率で指定する方式に変更するため、平均からの距離をZscoreに変更（平均０/標準偏差１のスケールに正規化する）
        data['zscore'] = (data['price'] - data['sma']) / data['price'].rolling(SMA).std()
        # data.dropna(inplace=True)
        
        # 標準偏差の倍率で指定する形式に変更（例：threshold=１を指定すると、価格が平均から標準偏差分を超えたら発動する形になる。）
        # sell signals (change each recored with -1 or Nan)
        data['position'] = np.where(data['zscore'] > threshold, -1, np.nan)
        # buy signals (again, change each record with 1 or copy same data: -1 or Nan)
        data['position'] = np.where(data['zscore'] < -threshold, 1, data['position'])
        # data['position'] = np.where(data['distance'] > threshold, -1, np.nan) 
        # data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])

        # crossing of current price and SMA (again, change each record with 0 or copy same data -1/1/Nan -> finally, all recored consist of -1/1/0/Nan)
        data['position'] = np.where(data['zscore'] * data['zscore'].shift(1)
                                     < 0, 0, data['position'])
        # data['position'] = np.where(data['distance'] *
        #                             data['distance'].shift(1) <0,
        #                             0, data['position'])
        data['position'] = data['position'].ffill().fillna(0) # Nanのレコードは前方のレコードの値を埋め込む。先頭付近の前方がないNanは０に置き換える。
        
        data['strategy'] = data['position'].shift(1) * data['return']
        # determine when a trade takes place
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

        # For debugging
        trades = (data['position'].diff().fillna(0) != 0).sum()
        print("SMA & threshold:", SMA, threshold)
        print("Number of trades:", trades)
        print("Absolute performance:", aperf)

        return round(aperf, 2), round(operf, 2)
    
    def update_and_run(self, params):
        SMA = int(params[0])
        if SMA < 1:
            SMA = 1  # SMA=0 や SMA<0 を完全に防ぐ
        threshold = float(params[1])
        return -self.run_strategy(SMA, threshold)[0]

    # def update_and_run(self, params):
    #     return -self.run_strategy(params[0], params[1])[0]

    def optimize_parameters(self, SMA_range, threshold_range):
        opt = brute(self.update_and_run, (SMA_range, threshold_range),  finish=None)
        best_SMA = int(opt[0])
        best_threshold = float(opt[1])
        return (best_SMA, best_threshold), -self.update_and_run([best_SMA, best_threshold])



if __name__ == '__main__':
    mrbt = MRVectorBacktester('GDX', '2010-1-1', '2020-12-31',
                                10000, 0.0)
    print(mrbt.run_strategy(SMA=25, threshold=5))
    mrbt = MRVectorBacktester('GDX', '2010-1-1', '2020-12-31',
                                10000, 0.001)
    print(mrbt.run_strategy(SMA=25, threshold=5))
    mrbt = MRVectorBacktester('GLD', '2010-1-1', '2020-12-31',
                                10000, 0.001)
    print(mrbt.run_strategy(SMA=42, threshold=7.5))