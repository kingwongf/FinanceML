import pandas as pd
import numpy as np
import pickle
import collections
import h5py




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

y_np = Xy['bin'].values
X_np = Xy.drop(columns=['bin']).values


print(len(Xy))
X_train, X_test = X[:idx_split], X[idx_split:]
y_train, y_test = y[:idx_split], y[idx_split:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

test_X_scaled = scaler.fit_transform(X_test)
test_y_scaled = scaler.fit_transform(y_test.reshape(-1, 1))
test_X_array, test_y_array = np.array(test_X_scaled), np.array(test_y_scaled)

train_y_scaled= scaler.fit_transform(y_train.reshape(-1, 1))
train_y_array= np.array(train_y_scaled)

## for model to read, we need array form of X [time span, number of time step to predict, no. of feat]

test_features_set = np.reshape(test_X_scaled, (test_X_scaled.shape[0], 1, test_X_scaled.shape[1]))
from tensorflow.keras.models import load_model
model = load_model("scripts/research/generalized_LSTM/generalized_LSTM.h5")

test_predictions = model.predict(test_features_set)
test_predictions = scaler.inverse_transform(test_predictions)
import matplotlib.pyplot as plt

print(len(np.append(train_y_array,test_y_array)))
print(len(np.append(np.zeros(idx_split),test_predictions)))

f, ax1 = plt.subplots(figsize=(11,5))
ax1.plot(Xy['EURTRY 4. close'].values, 'b')
ax2 = ax1.twinx()
ax2.plot(np.append(np.zeros(idx_split),test_predictions), 'r', linewidth=0.5)
ax2.plot(np.append(train_y_array,test_y_array), 'g', linewidth=0.5)
plt.title('test LSTM')
plt.legend()
plt.show()

