import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
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
readin_date = "2019-07-31"
#readin_date = str(today)
date_dir = "data/" + readin_date + "/"
date_parser = pd.to_datetime

# price.index = pd.to_datetime(price['date'], dayfirst=True)

source_latest = pd.read_pickle("data/open_closes_source_latest_2019-09-29.pkl").sort_index()

print(source_latest.index)