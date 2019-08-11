import pandas as pd
import numpy as np
import pickle

X = pd.read_pickle("data/generalized_LSTM/feat_generalized_LSTM.pkl")

y_1 = pickle.load(open("data/generalized_LSTM/x1_labels.pkl", 'rb'))

#print(y_1)

ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
#print(ratios[].columns)
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)
y1_closes_names = [close[2:17] for close in ratios]

for i, y in enumerate(y_1):
    print(i)
    print(y1_closes_names[i])
    print(y.count())

#print(x1_closes_ratios)

### TODO:  test with first fx, 'EURTRY 4. close'
###        in the x1_closes_ratios as it has most labels (2951)



Xy = pd.merge_asof(X.dropna().sort_index(), y_1[39]['bin'].sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms'))
print(Xy.count())
Xy['bin'] = Xy['bin'].fillna(0)
y = Xy['bin'].values
X = Xy.drop(columns=['bin']).values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

X_array, y_array = np.array(X), np.array(y)


## for model to read, we need array form of X [time span, number of time step to predict, no. of feat]

features_set = np.reshape(X_array, (X_array.shape[0], 1, X_array.shape[1]))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=40)


model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], X_array.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=1000, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error'
              , metrics=['accuracy', 'precision', 'recall'])
model.fit(features_set, y_array, epochs = 1000, batch_size = 32
          , validation_split=0.33, callbacks=[early_stopping])


model.save("scripts/research/generalized_LSTM/generalized_LSTM.h5")
