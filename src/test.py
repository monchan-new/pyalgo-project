
import numpy as np
import pandas as pd
from pylab import mpl, plt  
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
import random



raw = pd.read_csv('https://hilpisch.com/pyalgo_eikon_eod_data.csv',
                           index_col=0, parse_dates=True).dropna()



symbol = 'EUR='

data = pd.DataFrame(raw[symbol])

data.rename(columns={symbol: 'price'}, inplace=True)

data['return'] = np.log(data['price'] /
                        data['price'].shift(1))

data['direction'] = np.where(data['return'] > 0, 1, 0)

lags = 5

cols= []
for lag in range(1, lags + 1):
  col = f'lag_{lag}'
  data[col] = data['return'].shift(lag)
  cols.append(col)
data.dropna(inplace=True)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

optimizer = Adam(learning_rate=0.0001)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)

set_seeds()
model = Sequential()
model.add(Dense(64, activation='relu',
                input_shape=(lags,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

cutoff = '2017-12-31'

training_data = data[data.index < cutoff].copy()

mu, std = training_data.mean(), training_data.std()

training_data_ = (training_data - mu) / std

test_data = data[data.index >= cutoff].copy()

test_data_ = (test_data -mu) / std

model.fit(
    training_data[cols],
    training_data['direction'],
    epochs=50,
    verbose=False,
    validation_split=0.2,
    shuffle=False
)


res = pd.DataFrame(model.history.history)

res[['accuracy', 'val_accuracy']].plot(figsize=(10,6), style='--')

model.evaluate(training_data_[cols], training_data['direction'])

pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)

training_data['prediction'] = np.where(pred > 0, 1, -1)

training_data['strategy'] = (training_data['prediction'] *
                             training_data['return'])

training_data[['return', 'strategy']].sum().apply(np.exp)

training_data[['return', 'strategy']].cumsum(
).apply(np.exp).plot(figsize=(10,6));

model.evaluate(test_data_[cols], test_data['direction'])

pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)


test_data['prediction'] = np.where(pred > 0, 1, -1)

test_data['prediction'].value_counts()

test_data['strategy'] = (test_data['prediction'] *
                         test_data['return'])


test_data[['return', 'strategy']].sum().apply(np.exp)


test_data[['return', 'strategy']].cumsum(
).apply(np.exp).plot(figsize=(10,6))


data['momentum'] = data['return'].rolling(5).mean().shift(1)


data['volatility'] = data['return'].rolling(20).std().shift(1)


data['distance'] = (data['price'] -
                    data['price'].rolling(50).mean()).shift(1)


data.dropna(inplace=True)


cols.extend(['momentum', 'volatility', 'distance'])



training_data = data[data.index < cutoff].copy()


mu, std = training_data.mean(), training_data.std()


training_data_ = (training_data - mu) / std


test_data = data[data.index >= cutoff].copy()


test_data_ = (test_data - mu) / std


set_seeds()

optimizer = Adam(learning_rate=0.0001)
model = Sequential()
model.add(Dense(32, activation='relu',
                input_shape=(len(cols),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(training_data_[cols], training_data['direction'],
verbose=False, epochs=25)


model.evaluate(training_data_[cols], training_data['direction'])


pred = np.where(model.predict(training_data_[cols]) > 0.5, 1, 0)


training_data['prediction'] = np.where(pred > 0, 1, -1)


training_data['strategy'] = (training_data['prediction'] *
                             training_data['return'])


training_data[['return', 'strategy']].sum().apply(np.exp)


training_data[['return', 'strategy']].cumsum(
).apply(np.exp).plot(figsize=(10,6));


model.evaluate(test_data_[cols], test_data['direction'])


pred = np.where(model.predict(test_data_[cols]) > 0.5, 1, 0)
# pred = model.predict(test_data_[cols]).flatten()



test_data['prediction'] = np.where(pred > 0, 1, -1)
# test_data['prediction'] = np.where(
#     pred > 0.55, 1,          # 強い上昇 → Long
#     np.where(pred < 0.45, -1, 0)   # 強い下降 → Short、その他はノートレード
# )



test_data['prediction'].value_counts()


test_data['strategy'] = (test_data['prediction'] *
                             test_data['return'])


test_data[['return', 'strategy']].sum().apply(np.exp)


test_data[['return', 'strategy']].cumsum(
).apply(np.exp).plot(figsize=(10,6));




