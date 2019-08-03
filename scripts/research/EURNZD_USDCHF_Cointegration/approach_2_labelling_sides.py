import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

date = "2019-07-28"

## Approach 2 Cumsum filter and getEvents from Marcos to label the sides of the bet
## then we use a ML model to learn the sides of the bet
## first consider EURNZD

closes  = pd.read_pickle("scripts/research/EURNZD_USDCHF_Cointegration/EURNZD_USDCHF.pkl")


## TODO resample to dt mins
dt = '30T'


closes = closes.resample(dt).last()

closes.index = pd.to_datetime(closes.index, dayfirst=True)

closes['ratio'] = closes['EURNZD 4. close']/closes['USDCHF 4. close']

## get tEvents according to ratio
h=0.0001
tEvents = labelling_Marcos.getTEvents(closes['ratio'], h)

## tEvents of EURNZD, USDCHF and ratio
closes['tEvents_EURNZD'] = closes['EURNZD 4. close'].loc[tEvents]
closes['tEvents_USDCHF'] = closes['USDCHF 4. close'].loc[tEvents]
closes['tEvents_ratio'] = closes['ratio'].loc[tEvents]

## plot tEvents
fig, axs = plt.subplots(3,1, figsize=(30, 10), sharex=True)
closes['EURNZD 4. close'].plot(ax=axs[0])
closes['tEvents_EURNZD'].plot(ax=axs[0], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
closes['USDCHF 4. close'].plot(ax=axs[1])
closes['tEvents_USDCHF'].plot(ax=axs[1], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
closes['ratio'].plot(ax=axs[2])
closes['tEvents_ratio'].plot(ax=axs[2], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
#plt.show()
plt.close()


## apply triple barriers method to closing prices triggered by tEvents of pair trading ratio
maxHold = 5 ## dt*maxHold in min
t1 = labelling_Marcos.addVerticalBarrier(tEvents, closes['EURNZD 4. close'], numDays=maxHold)
minRet = 0.001
ptSl= [1,1]         ## upper barrier = trgt*ptSl[0] and lower barrier = trgt*ptSl[1]
trgt = labelling_Marcos.getDailyVol(closes['EURNZD 4. close'])  ## unit width of the horizon barrier


events = labelling_Marcos.getEvents(closes['EURNZD 4. close'], tEvents, ptSl, trgt, minRet, 1, t1)
labels = labelling_Marcos.getBins(events, closes['EURNZD 4. close'])


labelPlot = pd.merge_asof(closes['EURNZD 4. close'],labels,
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))


labelPlot.loc[labelPlot['bin'] == 1.0, 'bin_pos'] = labelPlot['EURNZD 4. close']
labelPlot.loc[labelPlot['bin'] == -1.0, 'bin_neg'] = labelPlot['EURNZD 4. close']


f, ax = plt.subplots(figsize=(11,5))

labelPlot['EURNZD 4. close'].plot(ax=ax, alpha=.5, label='close')
labelPlot['bin_pos'].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='buy', color='g')
labelPlot['bin_neg'].plot(ax=ax,ls='',marker='v', markersize=7,
                       alpha=0.75, label='sell', color='r')

ax.legend()
plt.title("%s min max holding period long and short signals for EURNZD"%(maxHold*int(dt[:-1])) + date )
#plt.savefig("resources/%s min max holding period long and short signals for EURNZD"%(maxHold*int(dt[:-1])) + date )
#plt.show()
plt.close()


## now we need to set up and train a ML model to predict the label/ side of the bet
## features: ratio
## label: side [-1, 0, 1], [short, not trade, long]

X = closes[['ratio','EURNZD 4. close']]
labelPlot['bin'] = labelPlot['bin'].fillna(0)
y = labelPlot['bin']

Xy = (pd.merge_asof(X, y,
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna())

X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values



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
clearly the model is insufficient as we have only 237 X points to train and validate with,
we will come back to this late when we have more data and eliminate weekends
'''








