import numpy as np
import pandas as pd
import tpqoa

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from scipy.optimize import brute


class AdaBoostVectorBacktester:
    """
    NN バックテスターと同じ構造で作った
    AdaBoost + DecisionTree 用バックテスター
    """

    def __init__(self, symbol, start, end, amount,
                 # tc=0.00010599557439495706,
                 tc=0.000025,
                 granularity='M10',
                 n_estimators=15,
                 max_depth=2,
                 min_samples_leaf=15,
                 random_state=100,
                 cfg_path='/workspace/src/pyalgo_netting.cfg'):

        # パラメータ
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.granularity = granularity

        # AdaBoost パラメータ
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # API
        self.api = tpqoa.tpqoa(cfg_path)

        # データ格納
        self.data = None
        self.data_subset = None
        self.feature_columns = None
        self.results = None

        # データ取得
        self.get_data()


    # -----------------------------
    # 1. データ取得
    # -----------------------------
    def get_data(self):
        raw = self.api.get_history(
            instrument=self.symbol,
            start=self.start,
            end=self.end,
            granularity=self.granularity,
            price='M'
        )

        raw = raw[raw['complete'] == True]
        raw.index = pd.to_datetime(raw.index)

        # print("BT raw index head:", raw.index[:10])
        # print("BT raw index tail:", raw.index[-10:])
        # print("BT raw shape:", raw.shape)

        raw.rename(columns={'c': 'price'}, inplace=True)
        raw = pd.DataFrame(raw['price'])

        raw['returns'] = np.log(raw['price'] / raw['price'].shift(1))
        self.data = raw

        # print(raw)


    # # -----------------------------
    # # 2. データ選択
    # # -----------------------------
    # def select_data(self, start, end):
    #     return self.data[(self.data.index >= start) &
    #                      (self.data.index <= end)].copy()


    # -----------------------------
    # 4. 特徴量準備
    # -----------------------------
    def prepare_features_all(self, 
                             momentum, 
                             sma,
                             volatility,
                             min_window,max_window,
                             lags):
        # df = self.data.copy()

        # rolling
        self.data['vol'] = self.data['returns'].rolling(volatility).std()
        self.data['mom'] = np.sign(self.data['returns'].rolling(momentum).mean())
        self.data['sma'] = self.data['price'].rolling(sma).mean()
        self.data['min'] = self.data['price'].rolling(min_window).min()
        self.data['max'] = self.data['price'].rolling(max_window).max()


        # print(self.data)


        df = self.data.copy()
        # lag
        for f in ['returns', 'vol', 'mom', 'sma', 'min', 'max']:
            for lag in range(1, lags + 1):
                df[f"{f}_lag_{lag}"] = df[f].shift(lag)

        df.dropna(inplace=True)

        # print(df)

        return df

    # -----------------------------
    # 6. バックテスト実行
    # -----------------------------
    def run_strategy(self,
                start_in, end_in,
                start_out, end_out,
                lags=6,
                momentum=5,
                sma=20,
                volatility=20,
                min_window=14,
                max_window=14):

        
        # print(self.data)
        
        # Optimizer用に保存
        self.start_in  = start_in
        self.end_in    = end_in
        self.start_out = start_out
        self.end_out   = end_out

        # ★ Plot用に保存
        self.lags = lags
        self.momentum = momentum
        self.sma = sma
        self.volatility = volatility
        self.min_window = min_window
        self.max_window = max_window

        # ★ 内部状態だけ初期化（self.data は初期化しない）
        self.data_subset = None
        self.feature_columns = None
        self.results = None
        self.model = None

        # 全期間の特徴量を作る
        df = self.prepare_features_all(
            momentum=momentum,
            sma=sma,
            volatility=volatility,
            min_window=min_window,
            max_window=max_window,
            lags=lags
        )

        
        # lag 展開後の特徴量だけを使う列を定義
        self.feature_columns = [
            col for col in df.columns
            if "lag_" in col
        ]

        # print(df[self.feature_columns])

        # train / test を日付で切る
        train = df.loc[start_in:end_in].copy()
        test  = df.loc[start_out:end_out].copy()

        # 標準化の基準は train から取る
        X_train = train[self.feature_columns]
        self.mu = X_train.mean()
        self.std = X_train.std()
        X_train = (X_train - self.mu) / self.std

        # print(train,X_train)

        y_train = np.where(train['returns'] > 0, 1, -1)

        # ⑤ モデル定義＆学習
        dtc = DecisionTreeClassifier(
            random_state=self.random_state,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        self.model = AdaBoostClassifier(
            estimator=dtc,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )

        
        self.model.fit(X_train, y_train)
        # print(X_train,y_train)

        

        # テスト側も同じ特徴量＋同じ標準化
        X_test = test[self.feature_columns]
        X_test = (X_test - self.mu) / self.std

        # 予測
        test['prediction'] = self.model.predict(X_test)

        # print(X_test, test['prediction'])

        # 戦略リターン
        test['strategy'] = (
            test['prediction'] * test['returns']
        )
        
        # print('rerurns, strategy=', test[['returns', 'strategy']].sum().apply(np.exp))
        
        # コスト
        trades = test['prediction'].diff().fillna(0) != 0
        test.loc[trades, 'strategy'] -= self.tc


        # print('strategy_tc=', test[['strategy']].sum().apply(np.exp))
        

        # 累積リターン
        test['creturns'] = (
            self.amount * np.exp(test['returns'].cumsum())
        )
        test['cstrategy'] = (
            self.amount * np.exp(test['strategy'].cumsum())
        )


        # ⑧ 結果を保存
        self.data_subset = test
        self.results = test

        aperf = self.results['cstrategy'].iloc[-1]
        operf = aperf - self.results['creturns'].iloc[-1]

        print("aperf=", aperf)

        return round(aperf, 2), round(operf, 2)


    # -----------------------------
    # 7. プロット
    # -----------------------------
    def plot_results(self):
        if self.results is None:
            print("Run a strategy first.")
            return

        title = (
            f"{self.symbol} | AdaBoost Strategy | "
            f"lags={self.lags} "
            f"momentum={self.momentum} "
            f"sma={self.sma} "
            f"vol={self.volatility} "
            f"min={self.min_window} "
            f"max={self.max_window}"
        )

        self.results[['creturns', 'cstrategy']].plot(figsize=(10, 6), title=title)




    def update_and_run(self, params):
        lags        = int(params[0])
        momentum    = int(params[1])
        sma         = int(params[2])
        volatility  = int(params[3])
        min_window  = int(params[4])
        max_window  = int(params[5])

        
        print('lags=',lags,'mom=',momentum, "sma=",sma,
               "volatility=",volatility, "min=",min_window, "max=", max_window)
        
        perf = self.run_strategy(
            self.start_in, self.end_in,
            self.start_out, self.end_out,
            lags=lags,
            momentum=momentum,
            sma=sma,
            volatility=volatility,
            min_window=min_window,
            max_window=max_window
        )

        return -perf[0]

    def optimize_parameters(self,
                        lags_range=None,
                        momentum_range=None,
                        sma_range=None,
                        volatility_range=None,
                        min_range=None,
                        max_range=None):

        default_ranges = {
            "lags":        (3, 13, 3),     # 3,6,9,12
            "momentum":    (5, 22, 8),     # 5,13,21
            "sma":         (10, 41, 15),   # 10,25,40
            "volatility":  (10, 31, 10),   # 10,20,30
            "min_window":  (14, 15, 100),  # 固定値 14
            "max_window":  (14, 15, 100),  # 固定値 14
        }

        ranges = [
            lags_range       if lags_range       is not None else default_ranges["lags"],
            momentum_range   if momentum_range   is not None else default_ranges["momentum"],
            sma_range        if sma_range        is not None else default_ranges["sma"],
            volatility_range if volatility_range is not None else default_ranges["volatility"],
            min_range        if min_range        is not None else default_ranges["min_window"],
            max_range        if max_range        is not None else default_ranges["max_window"],
        ]

        opt = brute(self.update_and_run, ranges, finish=None)

        best_params = opt
        best_perf = -self.update_and_run(opt)

        return best_params, best_perf
