import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

X = pd.read_pickle("data/generalized_LSTM/feat_generalized_LSTM.pkl")

y_1 = pickle.load(open("data/generalized_LSTM/x1_labels.pkl", 'rb'))

#print(y_1)

ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
#print(ratios[].columns)
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)
y1_closes_names = [close[2:17] for close in ratios]



### TODO:  test with first fx, 'EURTRY 4. close'
###        in the x1_closes_ratios as it has most labels (2951)



Xy = pd.merge_asof(X.dropna().sort_index(), y_1[39]['bin'].sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms'))

idx_split = Xy.index.get_loc('2019-08-01 00:00:00')
print(idx_split)
Xy['bin'] = Xy['bin'].fillna(0)
y = Xy['bin'].values
X = Xy.drop(columns=['bin']).values

'''
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=1)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
'''

print(len(Xy))
X_train, X_test = X[:idx_split], X[idx_split:]
y_train, y_test = y[:idx_split], y[idx_split:]

print(len(X_train), len(X_test))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

train_X_scaled, test_X_scaled = scaler.fit_transform(X_train), scaler.fit_transform(X_test)
train_y_scaled, test_y_scaled = scaler.fit_transform(y_train.reshape(-1, 1)), scaler.fit_transform(y_test.reshape(-1, 1))
train_X_array, train_y_array, test_X_array, test_y_array = np.array(train_X_scaled), np.array(train_y_scaled), np.array(test_X_scaled), np.array(test_y_scaled)


## for model to read, we need array form of X [time span, number of time step to predict, no. of feat]

features_set = np.reshape(train_X_scaled, (train_X_scaled.shape[0], 1, train_X_scaled.shape[1]))
test_features_set = np.reshape(test_X_scaled, (test_X_scaled.shape[0], 1, test_X_scaled.shape[1]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=40)


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], train_X_array.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=1000, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error'
              , metrics=['accuracy', 'mae'])
model.fit(features_set, train_y_array, epochs = 1000, batch_size = 32
          , validation_split=0.33, callbacks=[early_stopping])


model.save("scripts/research/generalized_LSTM/generalized_LSTM.h5")



test_predictions = model.predict(test_X_array)

test_predictions = scaler.inverse_transform(test_predictions)

plt.figure(figsize=(10,6))
plt.plot(X['EURTRY 4. close']['2019-08-01 00:00:00':'2019-08-08 02:18:00'], color='blue', label='EURTRY')
plt.plot(test_predictions, color='red', label='signals')
plt.title('test LSTM')
plt.legend()
plt.show()