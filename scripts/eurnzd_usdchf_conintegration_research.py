import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import featGen
import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

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

## Recap
''' from cointegration vcv research we know EURNZD and USDCHF are cointegrated

We want to develop a pairs trading strategy based on that, 
short EURNZD and long USDCHF when pairs ratio EURNZD/ USDCHF climbs
long EURNZD and short USDCHF when pairs ratio declines
'''

## resample to dt mins
dt = '30T'
closes = closes.resample(dt).last()


## plot historical ratio
pairs_trading_ratio = closes['EURNZD 4. close']/closes['USDCHF 4. close']
closes['ratio'] = pairs_trading_ratio
MA_1hr = featGen.ema(pairs_trading_ratio, 60)
MA_5hr = featGen.ema(pairs_trading_ratio, 60*5)
MA_10hr = featGen.ema(pairs_trading_ratio, 60*10)

fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(pairs_trading_ratio)
ax.plot(MA_1hr, 'r-')
ax.plot(MA_5hr, 'g-')
ax.plot(MA_10hr, 'b-')

plt.title("EURNZD / USDCHF Ratio" + date )
plt.savefig("resources/EURNZD to USDCHF Ratio on " + date +'.png' , dpi=fig.dpi)
plt.close()
''' Option 1 
Set upperbound and lowerbound using a +- expected vol
expected vol derived from GARCH
'''

## GARCH based on ret of pairs ratio
closes['ret'] = np.log(pairs_trading_ratio).diff()
closes['var'] = np.nan
lamb = 0.9
for i in range(len(closes)):
    if np.isnan(closes['ret'][i]):
        pass
    else:
        if i==1:
            closes['var'][i] = closes['ret'][i]**2
        else:
            closes['var'][i] = lamb*(closes['ret'][i-1]**2) + (1-lamb)*(closes['ret'][i-1]**2)


closes['vol'] = np.power(closes['var'],0.5)  # conditional vol
closes['mean'] = featGen.ema(closes['ret'], alpha=lamb)
closes['UB'] = closes['mean'] + 1.645*closes['vol']
closes['LB'] = closes['mean'] - 1.645*closes['vol']

closes['long USDCHF'] = closes['ret'] > closes['UB']
closes['short USDCHF'] = closes['ret'] < closes['LB']

closes['short EURNZD'] = closes['ret'] > closes['UB']
closes['long EURNZD']= closes['ret'] < closes['LB']


closes.loc[closes['long USDCHF'] == True, 'USDCHF_pos'] = closes['USDCHF 4. close']
closes.loc[closes['short USDCHF'] == True, 'USDCHF_neg'] = closes['USDCHF 4. close']

closes.loc[closes['long EURNZD'] == True, 'EURNZD_pos'] = closes['EURNZD 4. close']
closes.loc[closes['short EURNZD'] == True, 'EURNZD_neg'] = closes['EURNZD 4. close']



fig, axs = plt.subplots(4,1, figsize=(30, 10), sharex=True)



closes['EURNZD 4. close'].plot(ax=axs[0])
closes['EURNZD_pos'].plot(ax=axs[0], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
closes['EURNZD_neg'].plot(ax=axs[0],ls='',marker='v', markersize=7,
                       alpha=0.75, label='stop loss', color='r')

axs[0].set_title('EURNZD')

closes['USDCHF 4. close'].plot(ax=axs[1])
closes['USDCHF_pos'].plot(ax=axs[1], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')
closes['USDCHF_neg'].plot(ax=axs[1],ls='',marker='v', markersize=7,
                       alpha=0.75, label='stop loss', color='r')
axs[1].set_title('USDCHF')

closes['ret'].plot(ax=axs[2])
closes['UB'].plot(ax=axs[2], color='g')
closes['LB'].plot(ax=axs[2], color='r')
axs[2].set_title('Ratio Change with rolling UB and LB')

closes['ratio'].plot(ax=axs[3])
axs[3].set_title('Ratio')

#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#fig.subplots_adjust(hspace = .5, wspace=.001)
#plt.show()
plt.suptitle("Pairs Trading GARCH Boundaries Approach")
plt.savefig("resources/Pairs Trading GARCH Approach.png")


## Option 2 Cumsum filter from Marcos

tEvents = getTEvents(['ratio'], h)
'''
print(closes[['ratio','long', 'short']])

closes.loc[closes['long'] == 1.0, ''] = closes['EURNZD 4. close']
closes.loc[closes['short'] == -1.0, 'bin_neg'] = Xy['4. close']
'''