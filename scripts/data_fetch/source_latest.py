import numpy as np
import sklearn.covariance
import datetime
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


''' merge dataframes of different dates to have one large dataframe '''


tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

interval = "1min"
## range of dates
numdays=11

base = datetime.datetime.today()
datelist = [base - datetime.timedelta(days=x) for x in range(numdays)]
datelist = [x.strftime('%Y-%m-%d') for x in datelist]
#datelist = ["2019-08-04", "2019-08-03"]
print(datelist)



full_open_closes = pd.DataFrame([])
for t, readin_date in enumerate(datelist):

    open_closes = pd.DataFrame([])

    date_dir = "data/" + readin_date + "/"
    date_parser = pd.to_datetime
    #prices = [pd.read_csv("data/" + interval + '_price_' + ticker + "_" + str(today) + '.csv', date_parser=date_parser) for ticker in tickers]
    prices = [pd.read_csv( date_dir + interval + '_price_' + ticker + "_" + readin_date + '.csv', date_parser=date_parser) for ticker in tickers]



    for i,ticker in enumerate(tickers):
        prices[i].index = pd.to_datetime(prices[i]['date'], dayfirst=True)
        if i==0:
            open_closes = prices[i][[ticker + " 4. close", ticker + " 1. open"]]
        else:
            open_closes = pd.merge_asof(open_closes.dropna().sort_index(), prices[i][[ticker + " 4. close", ticker + " 1. open"]].sort_index(),
                        left_index=True, right_index=True,
                        direction='forward',tolerance=pd.Timedelta('2ms')).dropna()

    if t==0:
        full_open_closes = open_closes
    else:
        full_open_closes = full_open_closes.append(open_closes)


## needed to drop duplicates
full_open_closes =  full_open_closes.loc[~full_open_closes.index.duplicated(keep='first')]

#closes = closes.resample(dt).last()

if len(full_open_closes.columns) < 5:
    s = '_'
    full_open_closes.to_pickle("data/source_latest_" + s.join(tickers) + ".pkl")
else:
    full_open_closes.to_pickle("data/source_latest.pkl")




