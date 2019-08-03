import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tools import featGen
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

interval = "1min"
today = date.today()
date = "2019-07-31"
date = str(today)
date_dir = "data/" + date + "/"
date_parser = pd.to_datetime
#prices = [pd.read_csv("data/" + interval + '_price_' + ticker + "_" + str(today) + '.csv', date_parser=date_parser) for ticker in tickers]
prices = [pd.read_csv( date_dir + interval + '_price_' + ticker + "_" + date + '.csv', date_parser=date_parser) for ticker in tickers]

# price.index = pd.to_datetime(price['date'], dayfirst=True)


closes = pd.DataFrame([])

for i,ticker in enumerate(tickers):
    prices[i].index = pd.to_datetime(prices[i]['date'], dayfirst=True)
    if i==0:
        closes = prices[i][ticker + " 4. close"]
    else:
        closes = pd.merge_asof(closes, prices[i][ticker + " 4. close"],
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna()


# Compute the correlation matrix
corr = closes.corr()



# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 0, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .5})

yticks = closes.index
xticks = closes.index
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Empirical Correlation Matrix on " + date )
#plt.savefig("resources/Empirical Correlation Matrix on " + date +'.png' , dpi=f.dpi)
#plt.show()
plt.close()

#print(corr)
absCorr = corr.abs()

# extract upper triangle without diagonal with k=1 
sol = (absCorr.where(np.triu(np.ones(absCorr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)).to_frame()
sol['pairs'] = sol.index
sol = sol.set_index(np.arange(len(sol.index)))

#print(sol)

adfStats = []
for i in range(len(sol)):
    model = sm.regression.linear_model.OLS(closes[sol['pairs'][i][0]], closes[sol['pairs'][i][1]])
    results = model.fit()
    pairAdfStats = sm.tsa.stattools.adfuller(results.resid)
    adfStats.append(pairAdfStats)

sol['adfStats'] = adfStats
coIntegrate = [(abs(x[0]) > abs(x[4]['5%'])) for x in adfStats]
sol['cointegration'] = coIntegrate


cointegratedPairs = sol[coIntegrate]
cointegratedPairs = cointegratedPairs.reset_index()

#print(cointegratedPairs)
n_column_graphs = len(cointegratedPairs)//2
# n_column_graphs = 10
fig, axs = plt.subplots(2,n_column_graphs, figsize=(30, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for i in range(len(cointegratedPairs)):
    pairs_trading_ratio = closes[cointegratedPairs['pairs'][i][0]]/closes[cointegratedPairs['pairs'][i][1]]
    MA_1hr = featGen.ema(pairs_trading_ratio, 60)
    axs[i].plot(pairs_trading_ratio)
    axs[i].plot(MA_1hr)
    axs[i].set_title(cointegratedPairs['pairs'][i][0] + "/" + cointegratedPairs['pairs'][i][1])
plt.suptitle("Ratio of Cointegrated Forex Pairs")
#plt.savefig("resources/Ratio of Cointegrated Forex Pairs on " + date +'.png' , dpi=f.dpi)
#plt.show()


plt.close('all')

fig, axs = plt.subplots(2,n_column_graphs, figsize=(30, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

print(axs)

i=0
for ax_row in axs:
    for ax in ax_row:
        ax2 = ax.twinx()
        ts1 = closes[cointegratedPairs['pairs'][i][0]]
        ts2 = closes[cointegratedPairs['pairs'][i][1]]
        ax.plot(ts1)
        ax2.plot(ts2, 'r-')
        ax.set_title(cointegratedPairs['pairs'][i][0] + " and " + cointegratedPairs['pairs'][i][1])
        i+=1
plt.suptitle("Cointegrated Forex Pairs")
plt.savefig("resources/Cointegrated Forex Pairs on " + date +'.png' , dpi=f.dpi)


''' test for cointegration wuth top nth pairs'''

