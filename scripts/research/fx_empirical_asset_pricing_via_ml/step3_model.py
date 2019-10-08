import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import regularizers



pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

Xy = pd.read_pickle('data/fx_empirical_asset_pricing_via_ml/Xy.pkl')
Xy.dayofweek = Xy.dayofweek.astype(np.float64)
# print(Xy.info(verbose=True))

train, test = train_test_split(Xy, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
# print(len(train), 'train examples')
# print(len(val), 'validation examples')
# print(len(test), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 32 # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)


# for feature_batch, label_batch in train_ds.take(1):
#  # print('Every feature:', list(feature_batch.keys()))
#  numeric_cols = list(feature_batch.keys())
#  print(numeric_cols)


## TODO customize below
numeric_cols = Xy.columns.tolist()
numeric_cols = [e for e in numeric_cols if e not in ('ticker', 'target')]

feature_columns = []

# numeric cols, make sure everything is float
for header in numeric_cols:
  # feature_columns.append(feature_column.numeric_column(header, normalizer_fn=lambda x: (x - 3.0) / 4.2))
  feature_columns.append(feature_column.numeric_column(header))

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list('ticker', Xy.ticker.unique().tolist())
# thal_one_hot = feature_column.indicator_column(thal)
# feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


model = tf.keras.Sequential([
  feature_layer,
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu',
                activity_regularizer=regularizers.l1(0.001)),
  #  layers.Dropout(0.5),
   # layers.BatchNormalization(),
   # layers.Dense(64, activation='relu'),
  #  layers.BatchNormalization(),
  #  layers.Dropout(0.1),
  #  layers.Dense(8, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='linear')
    # layers.LeakyReLU(alpha=0.3)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[coeff_determination, 'accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5, callbacks=[callback])

print(model.layers[0].output)

loss, coeff_determination, accuracy = model.evaluate(test_ds)
print("R^2", coeff_determination)
print("Accuracy", accuracy)
