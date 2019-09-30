import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import moviepy
import matplotlib.ticker as matplotticker
from tools import featGen
from tools import clean_weekends
print(pd.__version__)

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
date_parser = pd.to_datetime

# price.index = pd.to_datetime(price['date'], dayfirst=True)

source_latest = pd.read_pickle("data/open_closes_source_latest_2019-09-29.pkl").sort_index()
source_latest_closes = source_latest[[col for col in source_latest.columns.tolist() if "close" in col]]
date_li = np.unique(source_latest_closes.index.date[source_latest_closes.index.dayofweek !=6]) ## excluding Sunday as it starts on 19:00

for i in range(len(date_li)-5):
    closes = source_latest_closes[date_li[i]:date_li[i+5]]
    corr = closes.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 20))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 0, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.8, cbar_kws={"shrink": .5})
    yticks = closes.index
    xticks = closes.index
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.xticks(rotation=90)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title("Empirical Correlation Matrix on " + str(date_li[i]))
    plt.savefig("resources/Full Empirical Correlation Matricies/%s Full Empirical Correlation Matrix on "%i+5 + str(date_li[i]) + '.png', dpi=f.dpi)
    plt.close()

import glob
import moviepy.editor as mpy

gif_name = 'Upper Tri Empirical Correlation Matrix'
fps = 6
file_list = glob.glob('resources/Empirical Correlation Matricies/*.png') # Get all the pngs in the current directory
print([int(x.split('/')[2][0:2]) for x in file_list])
list.sort(file_list, key=lambda x: int(x.split('/')[2][0:2])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)


####
'''
# print(source_latest.corr())
#print(source_latest.index)
# print(source_latest.rolling('1d').corr())




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
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Empirical Correlation Matrix on " + readin_date)
plt.savefig("resources/Empirical Correlation Matrix on " + readin_date +'.png' , dpi=f.dpi)
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
# n_column_graphs = len(cointegratedPairs)//2
n_column_graphs = 5
fig, axs = plt.subplots(2,n_column_graphs, figsize=(30, 10), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()


## to exclude weekend gaps

N, ind, date = clean_weekends.get_Nind(closes)
def format_date(x, pos=None):
    thisind = np.clip(int(x + 0.5), 0, N - 1)
    return date[thisind].strftime('%Y-%m-%d')

#for i in range(len(cointegratedPairs) - 1):
for i in range(10):
    pairs_trading_ratio = closes[cointegratedPairs['pairs'][i][0]]/closes[cointegratedPairs['pairs'][i][1]]
    MA_1hr = featGen.ema(pairs_trading_ratio, 60)
    axs[i].plot(ind, pairs_trading_ratio)
    axs[i].plot(ind, MA_1hr)
    axs[i].xaxis.set_major_formatter(matplotticker.FuncFormatter(format_date))
    axs[i].set_title(cointegratedPairs['pairs'][i][0] + "/" + cointegratedPairs['pairs'][i][1])
#fig.autofmt_xdate()
plt.suptitle("Ratio of Cointegrated Forex Pairs")
#plt.show()
plt.savefig("resources/Ratio of Cointegrated Forex Pairs on " + readin_date +'.png' , dpi=f.dpi)



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
        ax.plot(ind,ts1)
        ax2.plot(ind,ts2, 'r-')
        ax.xaxis.set_major_formatter(matplotticker.FuncFormatter(format_date))
        ax2.xaxis.set_major_formatter(matplotticker.FuncFormatter(format_date))
        ax.set_title(cointegratedPairs['pairs'][i][0] + " and " + cointegratedPairs['pairs'][i][1])
        i+=1
plt.suptitle("Cointegrated Forex Pairs")
plt.show()
plt.savefig("resources/Cointegrated Forex Pairs on " + readin_date +'.png' , dpi=f.dpi)

'''
''' test for cointegration wuth top nth pairs'''

