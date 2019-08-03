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

## Approach 2 Cumsum filter and getEvents from Marcos to label the sides of the bet
## first consider EURNZD


tickers = ['EURNZD', 'USDCHF']

interval = "1min"
today = date.today()
date = "2019-07-28"
date_dir = "data/" + date + "/"
date_parser = pd.to_datetime
#prices = [pd.read_csv("data/" + interval + '_price_' + ticker + "_" + str(today) + '.csv', date_parser=date_parser) for ticker in tickers]
prices = [pd.read_csv( date_dir + interval + '_price_' + ticker + "_" + date + '.csv', date_parser=date_parser) for ticker in tickers]



closes = pd.DataFrame([])

for i,ticker in enumerate(tickers):
    prices[i].index = pd.to_datetime(prices[i]['date'], dayfirst=True)
    if i==0:
        closes = prices[i][ticker + " 4. close"]
    else:
        closes = pd.merge_asof(closes, prices[i][ticker + " 4. close"],
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna()

## TODO resample to dt mins
dt = '1T'


closes = closes.resample(dt).last()

closes.index = pd.to_datetime(closes.index, dayfirst=True)

closes['ratio'] = closes['EURNZD 4. close']/closes['USDCHF 4. close']

## get tEvents according to ratio
h=0.001
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

"""
f,ax=plt.subplots()
trgt.plot(ax=ax)
ax.axhline(trgt.mean(),ls='--',color='r')
plt.show()
plt.close()
"""


events = labelling_Marcos.getEvents(closes['EURNZD 4. close'], tEvents, ptSl, trgt, minRet, 1, t1)
labels = labelling_Marcos.getBins(events, closes['EURNZD 4. close'])

Xy = pd.merge_asof(closes['EURNZD 4. close'],labels,
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))


Xy.loc[Xy['bin'] == 1.0, 'bin_pos'] = Xy['EURNZD 4. close']
Xy.loc[Xy['bin'] == -1.0, 'bin_neg'] = Xy['EURNZD 4. close']


f, ax = plt.subplots(figsize=(11,5))

Xy['EURNZD 4. close'].plot(ax=ax, alpha=.5, label='close')
Xy['bin_pos'].plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='buy', color='g')
Xy['bin_neg'].plot(ax=ax,ls='',marker='v', markersize=7,
                       alpha=0.75, label='sell', color='r')

ax.legend()
plt.title("%s min max holding period long and short signals for EURNZD"%(maxHold*int(dt[:-1])) + date )
#plt.savefig("resources/%s min max holding period long and short signals for EURNZD"%(maxHold*int(dt[:-1])) + date )
#plt.show()
plt.close()

