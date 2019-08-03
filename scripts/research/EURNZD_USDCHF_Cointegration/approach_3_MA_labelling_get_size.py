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
closes  = pd.read_pickle("scripts/research/EURNZD_USDCHF_Cointegration/EURNZD_USDCHF.pkl")
## TODO resample to dt mins
dt = '5T'


closes = closes.resample(dt).last()

closes.index = pd.to_datetime(closes.index, dayfirst=True)

closes['ratio'] = closes['EURNZD 4. close']/closes['USDCHF 4. close']

### Approach 3, using a MA crossover strategy to decide the side of the bet then a Neural Net to decide to trade or not

closes['MA_5'] = featGen.ema(closes['ratio'] , 5) ## 5*dt, 30mins
closes['MA_10'] = featGen.ema(closes['ratio'] , 10)
closes['MA_200'] = featGen.ema(closes['ratio'] , 200)

def get_up_cross(fast, slow):
    crit1 = fast.shift(1) < slow.shift(1) ## before
    crit2 = fast > slow
    return fast[(crit1) & (crit2)]

def get_down_cross(fast, slow):
    crit1 = fast.shift(1) > slow.shift(1)
    crit2 = fast < slow
    return fast[(crit1) & (crit2)]

up = get_up_cross(closes['MA_10'], closes['MA_200'])
down = get_down_cross(closes['MA_10'], closes['MA_200'])

f, ax = plt.subplots(2,1,figsize=(11,8))

closes['ratio'].plot(ax=ax[0], alpha=.5)
closes['MA_10'].plot(ax=ax[0], label='MA 10')
closes['MA_200'].plot(ax=ax[0], label='MA 200')
up.plot(ax=ax[0],ls='',marker='^', markersize=7,
                     alpha=0.75, label='upcross', color='g')
down.plot(ax=ax[0],ls='',marker='v', markersize=7,
                       alpha=0.75, label='downcross', color='r')
closes['EURNZD 4. close'].plot(ax=ax[1])
ax[1].set_title('Ratio')
ax[1].set_title('EURNZD close')
plt.show()
'''
closes['EURNZD_pos'].plot(ax=axs[0], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')

'''



'''
print(closes[['ratio','long', 'short']])

closes.loc[closes['long'] == 1.0, ''] = closes['EURNZD 4. close']
closes.loc[closes['short'] == -1.0, 'bin_neg'] = Xy['4. close']
'''