import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, tqdm_notebook
import mp
import labelling


price = pd.read_csv("daily_price_EURUSD_2019-06-27.csv")

price.index = pd.to_datetime(price['date'])



## apply triple barrier to get side of the bet of bins [-1, 0, 1]

## CUSUM filter to define events of deviaiting from mean exceeding thereshold
tEvents = labelling.getTEvents(price['4. close'], 0.1)

maxHold = 3
t1 = labelling.addVerticalBarrier(tEvents, price['4. close'], numDays=maxHold)
minRet = 0.008
ptSl= [1,1]
trgt = labelling.getDailyVol(price['4. close'])

""" f,ax=plt.subplots()
trgt.plot(ax=ax)
ax.axhline(trgt.mean(),ls='--',color='r')
plt.show() """

events = labelling.getEvents(price['4. close'], tEvents, ptSl, trgt, minRet, 1, t1)
labels = labelling.getBins(events, price['4. close'])

Xy = pd.merge_asof(price,labels,
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))


Xy.loc[Xy['bin'] == 1.0, 'bin_pos'] = Xy['4. close']
Xy.loc[Xy['bin'] == -1.0, 'bin_neg'] = Xy['4. close']


f, ax = plt.subplots(figsize=(11,8))

Xy['4. close'].plot(ax=ax, alpha=.5, label='close')
Xy['bin_pos'].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
Xy['bin_neg'].plot(ax=ax,ls='',marker='v', markersize=7,
                       alpha=0.75, label='stop loss', color='r')

ax.legend()
plt.title("Long only, %s day max holding period profit taking and stop loss exit"%maxHold)

plt.show()
