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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class NeuralNetworkVectorBacktester:
    '''
    Vectorized backtesting class for Keras-based neural network strategies.
    Similar structure to ScikitVectorBacktester.
    '''
    
    def __init__(self, symbol, start, end, amount, tc,
                 lags=5, feature_config=None,
                  threshold_long=0.55, threshold_short=0.45,
                  neurons=32, epochs=25, lr=0.0005):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.lags = lags
        self.feature_config = feature_config or {
            "lag": [1,2,3,4,5],
            "momentum": [],
            "volatility": [],
            "sma": [],
            "ema": [],
            "range": []
        }
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.neurons = neurons
        self.epochs = epochs
        self.lr = lr
        self.results = None
        
        self.get_data()

    # -----------------------------
    # 1. データ取得
    # -----------------------------
    def get_data(self):
        raw = pd.read_csv(
            'https://hilpisch.com/pyalgo_eikon_eod_data.csv',
            index_col=0, parse_dates=True
        ).dropna()

        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
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

        # ★ 標準化対象は特徴量のみ
        self.feature_columns = [
            col for col in data.columns if col not in ["price", "returns"]
        ]

        self.mu = data[self.feature_columns].mean()
        self.std = data[self.feature_columns].std()
        data[self.feature_columns] = (data[self.feature_columns] - self.mu) /self.std

        self.data_subset = data

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
        self.prepare_features(start, end)

        X = self.data_subset[self.feature_columns]
        y = np.where(self.data_subset['returns'] > 0, 1, 0)

        self.model = self.build_model(len(self.feature_columns))
        self.model.fit(X, y, epochs=self.epochs, verbose=False)

    # -----------------------------
    # 6. バックテスト実行
    # -----------------------------
    def run_strategy(self, start_in, end_in, start_out, end_out):
        self.fit_model(start_in, end_in)

        # テストデータ準備
        self.prepare_features(start_out, end_out)

        X = self.data_subset[self.feature_columns]

        # NN の確率出力
        pred_prob = self.model.predict(X).flatten()
        
        # threshold による売買シグナル
        self.data_subset['prediction'] = np.where(
            pred_prob > self.threshold_long, 1,
            np.where(pred_prob < self.threshold_short, -1, 0)
        )

        # 戦略リターン
        self.data_subset['strategy'] = (self.data_subset['prediction'] * 
                                   self.data_subset['returns'])

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
    
    # -----------------------------
    # 7. プロット
    # -----------------------------
    def plot_results(self):
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        title = f'{self.symbol} | NN Strategy | TC =  {self.tc:.4f}'
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10,6)) 
