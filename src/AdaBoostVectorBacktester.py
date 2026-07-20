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

        # 特徴量設定
        self.feature_config = None

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
        raw.rename(columns={'c': 'price'}, inplace=True)

        raw['returns'] = np.log(raw['price'] / raw['price'].shift(1))
        self.data = raw.dropna()


    # -----------------------------
    # 2. データ選択
    # -----------------------------
    def select_data(self, start, end):
        return self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()


    # -----------------------------
    # 3. 特徴量生成
    # -----------------------------
    def add_features(self, data):
        # momentum
        for k in self.feature_config.get("momentum", []):
            data[f"mom_{k}"] = data["price"] - data["price"].shift(k)

        # volatility
        for k in self.feature_config.get("volatility", []):
            data[f"vol_{k}"] = data["returns"].rolling(k).std()

        # SMA
        for k in self.feature_config.get("sma", []):
            data[f"sma_{k}"] = data["price"].rolling(k).mean()

        # # Range
        # for k in self.feature_config.get("range", []):
        #     data[f"range_{k}"] = (
        #         data["price"].rolling(k).max() - data["price"].rolling(k).min()
        #     )

        # Min
        for k in self.feature_config.get("min", []):
            data[f"min_{k}"] = data["price"].rolling(k).min()

        # Max
        for k in self.feature_config.get("max", []):
            data[f"max_{k}"] = data["price"].rolling(k).max()



    # -----------------------------
    # 4. 特徴量準備（標準化含む）
    # -----------------------------
    def prepare_features(self, start, end):
        data = self.select_data(start, end)

        self.add_features(data)
        data.dropna(inplace=True)

        # テスト期間の標準化（学習期間の mu/std を使う）
        if hasattr(self, "mu"):
            data[self.feature_columns] = (data[self.feature_columns] - self.mu) / self.std

        self.data_subset = data

    # -----------------------------
    #   縦方向の過去行を横に展開する
    # -----------------------------

    def make_lag_matrix(self, X, lags):
        lagged = []
        for i in range(1, lags + 1):
            lagged.append(X.shift(i))
        lagged = pd.concat(lagged, axis=1)

        # 列名を f"{col}_lag_{i}" にする
        lagged.columns = [
            f"{col}_lag_{i}"
            for i in range(1, lags + 1)
            for col in X.columns
        ]
        return lagged



    # -----------------------------
    # 5. モデル学習
    # -----------------------------
    def fit_model(self, start, end):

        # 特徴量生成
        self.prepare_features(start, end)

        # 特徴量列（return, vol, mom, sma, min, max）
        self.feature_columns = [
            col for col in self.data_subset.columns
            if col not in ["price", "returns"]
        ]

        # 標準化
        self.mu = self.data_subset[self.feature_columns].mean()
        self.std = self.data_subset[self.feature_columns].std()

        X = (self.data_subset[self.feature_columns] - self.mu) / self.std

        # ★ 過去 Lags 行を横に展開
        X = self.make_lag_matrix(X, self.lags)

        # 欠ける部分を除去
        valid = X.notna().all(axis=1)
        X = X[valid]

        self.data_subset = self.data_subset[valid]
        y = np.where(self.data_subset['returns'] > 0, 1, -1)

        # AdaBoost 学習
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
        self.model.fit(X, y)



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
                #  range_=14,
                 min_window=20,
                 max_window=20):


        # 特徴量設定
        self.feature_config = {
            # "lag": list(range(1, lags + 1)),  # ← これはもう不要
            "momentum":   [momentum],
            "volatility": [volatility],
            "sma":        [sma],
            # "range":      [range_],
            "min":        [min_window],
            "max":        [max_window],
        }
        self.lags = lags  # ← fit_model / make_lag_matrix で使うのでここでセット


        # 学習
        self.fit_model(start_in, end_in)

        # テスト特徴量生成
        self.prepare_features(start_out, end_out)
        
        X = self.data_subset[self.feature_columns]

        # ★ 過去 Lags 行を横に展開
        X = self.make_lag_matrix(X, self.lags)

        # shift(1)（未来リーク防止）
        X = X.shift(1)
        # X = self.data_subset[self.feature_columns].shift(1)

        valid = X.notna().all(axis=1)
        X = X[valid]
        self.data_subset = self.data_subset[valid]

        # 予測
        prediction = self.model.predict(X)
        self.data_subset['prediction'] = prediction

        # 戦略リターン
        self.data_subset['strategy'] = (
            self.data_subset['prediction'] * self.data_subset['returns']
        )

        # コスト
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        self.data_subset.loc[trades, 'strategy'] -= self.tc

        # 累積リターン
        self.data_subset['creturns'] = (
            self.amount * np.exp(self.data_subset['returns'].cumsum())
        )
        self.data_subset['cstrategy'] = (
            self.amount * np.exp(self.data_subset['strategy'].cumsum())
        )

        self.results = self.data_subset

        aperf = self.results['cstrategy'].iloc[-1]
        operf = aperf - self.results['creturns'].iloc[-1]
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
            f"mom={self.feature_config['momentum']} "
            f"sma={self.feature_config['sma']} "
            f"vol={self.feature_config['volatility']} "
            f"min={self.feature_config['min']} "
            f"max={self.feature_config['max']}"
        )


        self.results[['creturns', 'cstrategy']].plot(figsize=(10, 6), title=title)