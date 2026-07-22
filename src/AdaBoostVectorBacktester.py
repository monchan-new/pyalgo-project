import numpy as np
import pandas as pd
import tpqoa

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


class AdaBoostVectorBacktester:
    """
    NN バックテスターと同じ構造で作った
    AdaBoost + DecisionTree 用バックテスター
    """

    def __init__(self, symbol, start, end, amount,
                tc=0.00010599557439495706,
                #  tc=0.000025,
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


    # -----------------------------
    # 2. データ選択
    # -----------------------------
    def select_data(self, start, end):
        return self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()


    # -----------------------------
    # 4. 特徴量準備
    # -----------------------------
    def prepare_features_all(self, window, lags):
        # df = self.data.copy()

        # rolling
        self.data['vol'] = self.data['returns'].rolling(window).std()
        self.data['mom'] = np.sign(self.data['returns'].rolling(window).mean())
        self.data['sma'] = self.data['price'].rolling(window).mean()
        self.data['min'] = self.data['price'].rolling(window).min()
        self.data['max'] = self.data['price'].rolling(window).max()

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
                 window=20):
        
        # ★ 内部状態だけ初期化（self.data は初期化しない）
        self.data_subset = None
        self.feature_columns = None
        self.results = None
        self.model = None

        # print(self.data)

        self.lags = lags
        self.window = window

        # 全期間の特徴量を作る
        df = self.prepare_features_all(window, lags)
        
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

        # コスト
        trades = test['prediction'].diff().fillna(0) != 0
        test.loc[trades, 'strategy'] -= self.tc

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

        print(aperf)

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
            f"mom={self.window} "
            f"sma={self.window} "
            f"vol={self.window} "
            f"min={self.window} "
            f"max={self.window}"
        )


        self.results[['creturns', 'cstrategy']].plot(figsize=(10, 6), title=title)