#
# Python Module with Class
# for Vectorized Backtesting
# of Neural Network-Based Strategies
#
# Python for Algorithmic Trading
# (c) Shoji Inada
#
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tpqoa
from scipy.optimize import brute


class NeuralNetworkVectorBacktester:
    '''
    Vectorized backtesting class for Keras-based neural network strategies.
    Similar structure to ScikitVectorBacktester.
    '''
    
    def __init__(self, symbol, start, end, amount, tc=0.000025,
                  threshold_long=0.50, threshold_short=0.50,
                  neurons=32, epochs=25, lr=0.0005, granularity='D'):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.neurons = neurons
        self.epochs = epochs
        self.lr = lr

        self.granularity = granularity # example: 'D"

        self.results = None
        

        # API は1回だけ作る
        self.api = tpqoa.tpqoa('/workspace/src/pyalgo_netting.cfg')

        self.get_data()

    # -----------------------------
    # 1. データ取得
    # -----------------------------
    def get_data(self):
        
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

        # # --- ⑥ 期間でフィルタリング ---
        # raw = raw.loc[self.start:self.end]

        # raw = pd.read_csv(
        #     'https://hilpisch.com/pyalgo_eikon_eod_data.csv',
        #     index_col=0, parse_dates=True
        # ).dropna()
        # raw = pd.DataFrame(raw[self.symbol])
        # raw = raw.loc[self.start:self.end]
        # raw.rename(columns={self.symbol: 'price'}, inplace=True)

        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    # -----------------------------
    # 2. データ選択
    # -----------------------------
    def select_data(self, start, end):
        data = self.data[(self.data.index >= start) &
                         (self.data.index <= end)].copy()
        return data
    
    # -----------------------------
    # 3. 特徴量生成
    # -----------------------------
    def add_features(self, data):
        # lag
        for k in self.feature_config.get("lag", []):
            data[f"lag_{k}"] = data["returns"].shift(k)

        # momentum
        for k in self.feature_config.get("momentum", []):
            data[f"mom_{k}"] = data["price"] - data["price"].shift(k)

        # volatility
        for k in self.feature_config.get("volatility", []):
            data[f"vol_{k}"] = data["returns"].rolling(k).std()

        # SMA
        for k in self.feature_config.get("sma", []):
            data[f"sma_{k}"] = data["price"].rolling(k).mean()

        # EMA
        for k in self.feature_config.get("ema", []):
            data[f"ema_{k}"] = data["price"].ewm(span=k).mean()

        # Range
        for k in self.feature_config.get("range", []):
            data[f"range_{k}"] = (
                data["price"].rolling(k).max() - data["price"].rolling(k).min()
            )

    def prepare_features(self, start, end):
        data = self.select_data(start,end)

        # ★ 追加特徴量生成
        self.add_features(data)
        data.dropna(inplace=True)

        # ★ テスト期間のデータについてのみここで標準化するが、平均・標準偏差は学習期間で求めたもの、つまりmu,std は fit_model() で計算したものを使う
        if hasattr(self, "mu"):
            data[self.feature_columns] = (data[self.feature_columns] - self.mu) / self.std

        self.data_subset = data

        # print("prepare_features: rows =", len(data))


    # -----------------------------
    # 4. NN モデル構築
    # -----------------------------
    def build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(self.neurons, activation='relu', input_shape=(input_dim, )))
        model.add(Dense(self.neurons, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # -----------------------------
    # 5. モデル学習
    # -----------------------------
    def fit_model(self, start, end):
        # ★ JupyterLabの実行を想定して、前回の試行の値をリセット
        self.feature_columns = []
        if hasattr(self, "mu"): del self.mu
        if hasattr(self, "std"): del self.std
        self.model = None
        self.data_subset = None

        # まず特徴量を生成する
        self.prepare_features(start, end)

        # ★ prepare_features の後で feature_columns を決める
        self.feature_columns = [
            col for col in self.data_subset.columns
            if col not in ["price", "returns"]
        ]

        # ★ 学習期間で mu,std を計算（テキストと同じタイミング）
        self.mu = self.data_subset[self.feature_columns].mean()
        self.std = self.data_subset[self.feature_columns].std()

        # ★ 学習データを標準化（テキストと同じタイミング）
        X = (self.data_subset[self.feature_columns] - self.mu) / self.std


        # ★ １つ前のFeature を入力として、
        X = X.shift(1)
        # ★ 本日のリターン（をコード化したもの：1/0）をターゲットにする
        y = np.where(self.data_subset['returns'] > 0, 1, 0)

        # shift による NaN を除外
        valid = X.notna().all(axis=1)
        X = X[valid]
        y = y[valid]
        

        # print("fit_model: valid rows =", valid.sum())


        # 乱数シードをここで固定化し、計算結果がばらつかないようにする。
        random.seed(42)        
        np.random.seed(42)
        tf.random.set_seed(42)

        self.model = self.build_model(len(self.feature_columns))
        # テキスト同様にsplitとshuffleを追加した。
        self.model.fit(X, y, epochs=self.epochs, verbose=False, 
    validation_split=0.2, shuffle=False)
        # self.model.fit(X, y, epochs=self.epochs, verbose=False)


    # -----------------------------
    # 6. バックテスト実行
    # -----------------------------
    def run_strategy(self,
                 start_in, end_in,
                 start_out, end_out,
                 lags=3,
                 momentum=None,
                 sma=None,
                 ema=None,
                 volatility=None,
                 range_=None):

        # デフォルト Feature Range
        # 一つのFeatureについて複数の期間をリスト形式で指定できる。（例えば[10, 20]->sma10/sma20）しかし、実際には特徴量が多くなりすぎるため、Defaultとしては単一の値を設定している。
        default_ranges = {
            "momentum":   5,
            "sma":        20,
            "ema":        10,
            "volatility": 20,
            "range":      14,
        }

        # ★ None の場合はデフォルト値を使う（これが本来の姿）
        momentum   = momentum   if momentum   is not None else default_ranges["momentum"]
        sma        = sma        if sma        is not None else default_ranges["sma"]
        ema        = ema        if ema        is not None else default_ranges["ema"]
        volatility = volatility if volatility is not None else default_ranges["volatility"]
        range_     = range_     if range_     is not None else default_ranges["range"]

        feature_config = {
            "lag": list(range(1, lags + 1)), # lag配列をlagsの指定値に併せて自動生成
            "momentum":   [momentum],
            "volatility": [volatility],
            "sma":        [sma],
            "ema":        [ema],
            "range":      [range_],
        }

        self.lags = lags
        self.feature_config = feature_config
        self.fit_model(start_in, end_in)

        # テストデータ準備
        self.prepare_features(start_out, end_out)

        # ★ １つ前のFeature を使用する
        X = self.data_subset[self.feature_columns].shift(1)

        # shift による NaN を除外
        valid = X.notna().all(axis=1)
        X = X[valid]
        self.data_subset = self.data_subset[valid]


        # print("run_strategy: valid rows =", valid.sum())


        # NN の確率出力
        pred_prob = self.model.predict(X).flatten()
        
        # threshold による売買シグナル
        self.data_subset['prediction'] = np.where(
            pred_prob > self.threshold_long, 1,
            np.where(pred_prob < self.threshold_short, -1, 0)
        )

        # 戦略リターン
        # ★ PredictionはReturnの行にセットしてあるためShift(1)は不要。
        self.data_subset['strategy'] = self.data_subset['prediction'] * self.data_subset['returns']

        # トレードコスト
        trades = self.data_subset['prediction'].diff().fillna(0) != 0
        self.data_subset.loc[trades, 'strategy'] -= self.tc
        
        # 累積リターン
        self.data_subset['creturns'] = self.amount * np.exp(self.data_subset['returns'].cumsum())
        self.data_subset['cstrategy'] = self.amount * np.exp(self.data_subset['strategy'].cumsum())

        
        self.results = self.data_subset
        # absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
    

    def update_and_run(self, params):
        lags = int(params[0])
        momentum = int(params[1])
        sma = int(params[2])
        ema = int(params[3])
        volatility = int(params[4])
        range_ = int(params[5])

        print('lags=',lags,'mom=',momentum, "sma=",sma,
               "ema=",ema, "volatility=",volatility, "range=",range_)
        
        perf = self.run_strategy(
            self.start_in, self.end_in,
            self.start_out, self.end_out,
            lags=lags,
            momentum=momentum,
            sma=sma,
            ema=ema,
            volatility=volatility,
            range_=range_
        )
        print('performance=', perf)

        return -perf[0]
        # return -self.run_strategy(
        #     self.start_in, self.end_in,
        #     self.start_out, self.end_out,
        #     lags=lags,
        #     momentum=momentum,
        #     sma=sma,
        #     ema=ema,
        #     volatility=volatility,
        #     range_=range_
        # )[0]



    def optimize_parameters(self,
                        lags_range=None,
                        momentum_range=None,
                        sma_range=None,
                        ema_range=None,
                        volatility_range=None,
                        range_range=None,
                        start_in=None, end_in=None,
                        start_out=None, end_out=None):
        ''' usage:
        1. optimize_parameters(start_in, end_in, start_out, end_out) 
            →rangeはすべてDefaultを使用。
        2. optimize_parameters(momentum_range=(5, 16, 5), start_in, end_in, start_out, end_out) 
            →一部のrangeだけ指定して後はDefaultを使用。
        '''

        # 固定の日付をセット
        self.start_in  = start_in
        self.end_in    = end_in
        self.start_out = start_out
        self.end_out   = end_out

        # デフォルト Range
        default_ranges = {
            "lags":        (3, 12, 4),     # 3,7,11
            "momentum":    (5, 11, 5),     # 5,10
            "sma":         (20, 51, 30),   # 20,50
            "ema":         (10, 21, 10),   # 10,20
            "volatility":  (20, 21, 100),  # 20
            "range":       (14, 15, 100),  # 14
        }

        # rangeの指定がNoneであるパラメータにはデフォルトを使う
        ranges = [
            lags_range        if lags_range        is not None else default_ranges["lags"],
            momentum_range    if momentum_range    is not None else default_ranges["momentum"],
            sma_range         if sma_range         is not None else default_ranges["sma"],
            ema_range         if ema_range         is not None else default_ranges["ema"],
            volatility_range  if volatility_range  is not None else default_ranges["volatility"],
            range_range       if range_range       is not None else default_ranges["range"],
        ]

        # 6次元 brute
        opt = brute(self.update_and_run, ranges, finish=None)

        best_params = opt
        best_perf = -self.update_and_run(opt)

        return best_params, best_perf


    # -----------------------------
    # 7. プロット
    # -----------------------------
    def plot_results(self):
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        title = (
            f"{self.symbol} | NN Strategy | "
            f"lags={self.lags} | "
            f"mom={self.feature_config['momentum']} "
            f"sma={self.feature_config['sma']} "
            f"ema={self.feature_config['ema']} "
            f"vol={self.feature_config['volatility']} "
            f"range={self.feature_config['range']}"
        )
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10,6)) 
