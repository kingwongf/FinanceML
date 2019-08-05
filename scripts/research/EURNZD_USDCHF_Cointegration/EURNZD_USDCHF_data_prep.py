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



## Recap
''' from cointegration vcv research we know EURNZD and USDCHF are cointegrated

We want to develop a pairs trading strategy based on that, 
short EURNZD and long USDCHF when pairs ratio EURNZD/ USDCHF climbs
long EURNZD and short USDCHF when pairs ratio declines


We need to prepare and pickle the data first before running any of the 3 approaches

'''


tickers = ['EURNZD', 'USDCHF']

interval = "1min"

today = date.today()
readin_date = "2019-07-28"
date_dir = "data/" + readin_date + "/"
date_parser = pd.to_datetime
#prices = [pd.read_csv("data/" + interval + '_price_' + ticker + "_" + str(today) + '.csv', date_parser=date_parser) for ticker in tickers]
prices = [pd.read_csv( date_dir + interval + '_price_' + ticker + "_" + readin_date + '.csv', date_parser=date_parser) for ticker in tickers]



closes = pd.DataFrame([])

for i,ticker in enumerate(tickers):
    prices[i].index = pd.to_datetime(prices[i]['date'], dayfirst=True)
    if i==0:
        closes = prices[i][ticker + " 4. close"]
    else:
        closes = pd.merge_asof(closes, prices[i][ticker + " 4. close"],
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna()


closes = closes.loc[~closes.index.duplicated(keep='first')]
closes.to_pickle("scripts/research/EURNZD_USDCHF_Cointegration/EURNZD_USDCHF.pkl")


