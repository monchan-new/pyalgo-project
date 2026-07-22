# %%
import math
import time
import numpy as np
import pandas as pd
import datetime as dt
from pylab import plt, mpl

# %%
import tpqoa
api = tpqoa.tpqoa('/workspace/src/pyalgo_netting.cfg')

# %%
instrument = 'EUR_USD'

# %%
raw = api.get_history(instrument,
                      start='2020-06-08',
                      end='2020-06-13',
                      granularity='M10',
                      price='M')

# %%
spread = 0.00012
mean = raw['c'].mean()
ptc = spread /mean
ptc

# %%
raw['c'].plot(figsize=(10,6),legend=True)

# %%
data = pd.DataFrame(raw['c'])
data.columns = [instrument,]
window = 20
data['return'] = np.log(data / data.shift(1))
data['vol'] = data['return'].rolling(window).std()
data['mom'] = np.sign(data['return'].rolling(window).mean())
data['sma'] = data[instrument].rolling(window).mean()
data['min'] = data[instrument].rolling(window).min()
data['max'] = data[instrument].rolling(window).max()
data.dropna(inplace=True)


# %%
lags = 6
features = ['return', 'vol', 'mom', 'sma', 'min', 'max']
cols = []
for f in features:
    for lag in range(1, lags + 1):
        col = f'{f}_lag_{lag}'
        data[col] = data[f].shift(lag)
        cols.append(col)
data.dropna(inplace=True)

# %%
data['direction'] = np.where(data['return'] > 0, 1, -1)
data[cols].iloc[:lags, ]

# %%
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# %%
n_estimators = 15
random_state = 100
max_depth = 2
min_samples_leaf = 15
subsample = 0.33

# %%
dtc = DecisionTreeClassifier(random_state=random_state,
                             max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf)
model = AdaBoostClassifier(estimator=dtc,
                           n_estimators=n_estimators,
                           random_state=random_state)
split = int(len(data) * 0.7)
train = data.iloc[:split].copy()
# mu, std = train.mean(), train.std()
# train_ = (train - mu) / std
mu, std = train[cols].mean(), train[cols].std()
train_ = train.copy()
train_[cols] = (train[cols] - mu) / std


# %%
print("TEXT split index:", data.index[split])


# %%
model.fit(train_[cols], train['direction'])

# %%
accuracy_score(train['direction'], model.predict(train_[cols]))

# %%
test = data.iloc[split:].copy()
test_ = (test - mu) / std
test['position'] = model.predict(test_[cols])
accuracy_score(test['direction'], test['position'])

# %%
test['strategy'] = test['position'] * test ['return']
sum(test['position'].diff() != 0)

# %%
test['strategy_tc'] = np.where(test['position'].diff() != 0,
                               test['strategy'] - ptc,
                               test['strategy'])
test[['return', 'strategy', 'strategy_tc']].sum().apply(np.exp)

# %%
test[['return', 'strategy', 'strategy_tc']].cumsum().apply(np.exp).plot(figsize=(10,6))

print(test[['return', 'strategy', 'strategy_tc']].sum().apply(np.exp))
