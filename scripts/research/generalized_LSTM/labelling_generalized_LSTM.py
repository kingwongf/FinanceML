import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import pickle
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


'''need to run source_latest.py, cointegration_pairs_closes_source_latest.py first '''
open_closes = pd.read_pickle("data/source_latest.pkl")
open_closes.index = pd.to_datetime(open_closes.index, dayfirst=True)


ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)
x1_closes_ratios = [close[2:17] for close in ratios]
x2_closes_ratios = [close[21:36] for close in ratios]

h=0.0001 ## if set too high, tEvents won't be in trgt

cointegrated_tEvents= [labelling_Marcos.getTEvents(ratios[ratio], h) for ratio in ratios.columns]
#print(len(x_closes_ratios), len(ratios), len(cointegrated_tEvents))



## apply triple barriers method to closing prices triggered by tEvents of pair trading ratio
maxHold = 5 ## dt*maxHold in min
x1_cointegrated_t1 = [labelling_Marcos.addVerticalBarrier(cointegrated_tEvents[i], open_closes[x1_closes_ratios[i]],
                                                        numDays=maxHold) for i in range(len(x1_closes_ratios))]

x2_cointegrated_t1 = [labelling_Marcos.addVerticalBarrier(cointegrated_tEvents[i], open_closes[x2_closes_ratios[i]],
                                                       numDays=maxHold) for i in range(len(x2_closes_ratios))]
minRet = 0.001
ptSl= [1,1]         ## upper barrier = trgt*ptSl[0] and lower barrier = trgt*ptSl[1]



x1_trgt = [labelling_Marcos.getDailyVol(open_closes[close]) for close in x1_closes_ratios]
x2_trgt = [labelling_Marcos.getDailyVol(open_closes[close]) for close in x2_closes_ratios]

pd.DataFrame(x1_cointegrated_t1[0]).to_csv("data/generalized_LSTM/labelling/x1_t1.csv")


x1_events = [labelling_Marcos.getEvents(open_closes[x1_closes_ratios[i]], cointegrated_tEvents[i],
                                        ptSl, x1_trgt[i], minRet, 1,x1_cointegrated_t1[i]) for i in range(len(x1_closes_ratios))]

x2_events = [labelling_Marcos.getEvents(open_closes[x2_closes_ratios[i]], cointegrated_tEvents[i],
                                        ptSl, x2_trgt[i], minRet, 1, x2_cointegrated_t1[i]) for i in range(len(x2_closes_ratios))]

x1_labels = [labelling_Marcos.getBins(x1_events[i], open_closes[x1_closes_ratios[i]].sort_index()) for i in range(len(x1_closes_ratios))]

x2_labels = [labelling_Marcos.getBins(x2_events[i], open_closes[x2_closes_ratios[i]].sort_index()) for i in range(len(x2_closes_ratios))]


#print(len(x1_closes_ratios), len(x1_labels))
pickle.dump(x1_trgt, open("data/generalized_LSTM/labelling/x1_trgt.pkl", 'wb'))
pickle.dump(x2_trgt, open("data/generalized_LSTM/labelling/x2_trgt.pkl", 'wb'))
pickle.dump(x1_events, open("data/generalized_LSTM/labelling/x1_events.pkl", 'wb'))
pickle.dump(x2_events, open("data/generalized_LSTM/labelling/x2_events.pkl", 'wb'))
pickle.dump(x1_labels, open("data/generalized_LSTM/labelling/x1_labels.pkl", 'wb'))
pickle.dump(x2_labels, open("data/generalized_LSTM/labelling/x2_labels.pkl", 'wb'))

## TODO: plot to check if they are correctly labelled

#y_1 = pickle.load(open("data/generalized_LSTM/x1_labels.pkl", 'rb'))
'''
x1_labelPlot = pd.DataFrame(open_closes[x1_closes_ratios], index=open_closes.index)
print(x1_labelPlot.columns)
for i,label in enumerate(x1_labels):
    print(label)
    print(label.rename(columns=lambda x: x1_closes_ratios[i] + " " + x, inplace=True))
#    x1_labelPlot = pd.merge_asof(x1_labelPlot, label.rename(columns=lambda x: x1_closes_ratios[i] + " " + x, inplace=True),
#                  left_index=True, right_index=True, direction='forward'
#                   ,tolerance=pd.Timedelta('2ms'))

'''





## TODO: need to figure out how to handle multiple labels





'''
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

model.add(LSTM(units=50, .return_sequences=True, input_shape=(features_set.shape[1], X_array.shape[1])))
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




