import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

open_closes = pd.read_pickle("data/source_latest.pkl")

ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")

print(ratios)

h=0.00001
tEvents = labelling_Marcos.getTEvents(ratios, h)

print(len(tEvents))
'''
## apply triple barriers method to closing prices triggered by tEvents of pair trading ratio
maxHold = 5 ## dt*maxHold in min
t1 = labelling_Marcos.addVerticalBarrier(tEvents, closes['EURNZD 4. close'], numDays=maxHold)
minRet = 0.001
ptSl= [1,1]         ## upper barrier = trgt*ptSl[0] and lower barrier = trgt*ptSl[1]
trgt = labelling_Marcos.getDailyVol(closes['EURNZD 4. close'])  ## unit width of the horizon barrier


events = labelling_Marcos.getEvents(closes['EURNZD 4. close'], tEvents, ptSl, trgt, minRet, 1, t1)
labels = labelling_Marcos.getBins(events, closes['EURNZD 4. close'])



## now we need to set up and train a ML model to predict the label/ side of the bet
## features: ratio
## label: side [-1, 0, 1], [short, not trade, long]

labelPlot = pd.merge_asof(closes['EURNZD 4. close'],labels,
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))

X = closes[['ratio','EURNZD 4. close']]
labelPlot['bin'] = labelPlot['bin'].fillna(0)
y = labelPlot['bin']


Xy = (pd.merge_asof(X, y,
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna())

X = X.values
y = y.values




## Train RNN to predict label
## Data normalisation

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

X_array, y_array = np.array(X), np.array(y)



## for model to read, we need array form of X [time span, number of time step to predict, no. of feat]

features_set = np.reshape(X_array, (X_array.shape[0], 1, X_array.shape[1]))


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,shuffle=False)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


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
              , metrics=['accuracy'])
model.fit(features_set, y_array, epochs = 100, batch_size = 32
          , validation_split=0.33)

'''
''' 
clearly the model is insufficient as we have only 237 X points to train and validate with,
we will come back to this late when we have more data and eliminate weekends
'''




