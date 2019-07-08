import pandas as pd
import featGen
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import labelling_Marcos

from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split

from labelling_King import predict_ret




price = pd.read_csv("daily_price_EURUSD_2019-06-27.csv")

price.index = pd.to_datetime(price['date'], dayfirst=True)

K,D = featGen.stochRSI(price['4. close'], 14)
feat = (pd.DataFrame()
        .assign(ema5= featGen.ema(price['4. close'], 5))
        .assign(ema10=featGen.ema(price['4. close'], 10))
        .assign(ema200=featGen.ema(price['4. close'], 200))
        .assign(rsi=featGen.RSI(price['4. close'], 14))
        .assign(stoch_rsi_K= K)
        .assign(stoch_rsi_D = D)
        .assign(macd=featGen.MACD(price['4. close']))
        .drop_duplicates()
        .dropna()
        )

## apply triple barrier to get side of the bet of bins [-1, 0, 1]

## CUSUM filter to define events of deviaiting from mean exceeding thereshold
tEvents = labelling_Marcos.getTEvents(price['4. close'], 0.1)

maxHold = 3
t1 = labelling_Marcos.addVerticalBarrier(tEvents, price['4. close'], numDays=maxHold)
minRet = 0.00005
ptSl= [0, 2]
trgt = labelling_Marcos.getDailyVol(price['4. close'])

""" f,ax=plt.subplots()
trgt.plot(ax=ax)
ax.axhline(trgt.mean(),ls='--',color='r')
plt.show() """

events = labelling_Marcos.getEvents(price['4. close'], tEvents, ptSl, trgt, minRet, 1, t1)
# labels = labelling_Marcos.getBins(events, price['4. close'])

labels = predict_ret(price['4. close'], 3)
Xy = (pd.merge_asof(feat, labels[['bin']],
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna())

X = Xy.drop('bin',axis=1).values
y = Xy['bin'].values

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                    shuffle=False)


model = Sequential([
    Dense(32, input_shape=(7,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
