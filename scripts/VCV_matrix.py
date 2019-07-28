import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.width = 0

tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

interval = "1min"
# today = date.today()
date = "2019-07-27"
date_parser = pd.to_datetime
prices = [pd.read_csv("data/" + date + "/" + interval + '_price_'
                      + ticker + "_" + date + '.csv', date_parser=date_parser) for ticker in tickers]

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
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.8, cbar_kws={"shrink": .5})

yticks = closes.index
xticks = closes.index
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.gcf().subplots_adjust(bottom=0.15)
plt.title("Empirical Correlation Matrix on " + date )
plt.savefig("resources/" + "Empirical Correlation Matrix on " + date +'.png' , dpi=f.dpi)
plt.show()

