import pandas as pd
import numpy as np
import pickle

import h5py



X = pd.read_pickle("data/generalized_LSTM/feat_generalized_LSTM.pkl")
y_1 = pickle.load(open("data/generalized_LSTM/x1_labels.pkl", 'rb'))

Xy = pd.merge_asof(X.dropna().sort_index(), y_1[39]['bin'].sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms'))

test_features = Xy.drop(columns=['bin'])['2019-08-01 00:00:00':'2019-08-08 02:18:00'].values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

from tensorflow.keras.models import load_model

test_features_scaled =scaler.fit_transform(test_features)
test_features_array = np.array(test_features_scaled)

test_features_set = np.reshape(test_features_array, (test_features_array.shape[0], 1, test_features_array.shape[1]))
model = load_model("scripts/research/generalized_LSTM/generalized_LSTM.h5")

test_predictions = model.predict(test_features_set)

pickle.dump(test_predictions, open("data/generalized_LSTM/test_predicitions_generalized_LSTM.pkl", 'wb'))
test_predictions = scaler.inverse_transform(test_predictions)

print(test_predictions)