import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, tqdm_notebook
from tools import mp
from tools import labelling_Marcos


price = pd.read_csv("data/2019-07-27/1min_price_EURUSD_2019-07-27.csv")

price.index = pd.to_datetime(price['date'], dayfirst=True)



## apply triple barrier to get side of the bet of bins [-1, 0, 1]

## CUSUM filter to define events of deviaiting from mean exceeding thereshold
tEvents = labelling_Marcos.getTEvents(price['EURUSD 4. close'], 0.0001)

maxHold = 3
t1 = labelling_Marcos.addVerticalBarrier(tEvents, price['EURUSD 4. close'], numDays=maxHold)
minRet = 0.0008
ptSl= [1,1] ## upper barrier = trgt*ptSl[0] and lower barrier = trgt*ptSl[1]
trgt = labelling_Marcos.getDailyVol(price['EURUSD 4. close'])

""" f,ax=plt.subplots()
trgt.plot(ax=ax)
ax.axhline(trgt.mean(),ls='--',color='r')
plt.show() """

events = labelling_Marcos.getEvents(price['EURUSD 4. close'], tEvents, ptSl, trgt, minRet, 1, t1)

labels = labelling_Marcos.getBins(events, price['EURUSD 4. close'])

Xy = pd.merge_asof(price,labels,
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))


Xy.loc[Xy['bin'] == 1.0, 'bin_pos'] = Xy['EURUSD 4. close']
Xy.loc[Xy['bin'] == -1.0, 'bin_neg'] = Xy['EURUSD 4. close']


f, ax = plt.subplots(figsize=(11,8))

Xy['EURUSD 4. close'].plot(ax=ax, alpha=.5, label='close')
Xy['bin_pos'].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
Xy['bin_neg'].plot(ax=ax,ls='',marker='v', markersize=7,
                       alpha=0.75, label='stop loss', color='r')

ax.legend()
plt.title("Long only, %s day max holding period profit taking and stop loss exit"%maxHold)

plt.show()
