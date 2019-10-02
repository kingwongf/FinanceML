import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
from itertools import chain
from functools import reduce
from time import process_time
import swifter
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as matplotticker
from tools import featGen
import tools.featGen
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

source_latest_open_close = pd.read_pickle("data/open_closes_source_latest_2019-10-01.pkl").sort_index()
closes = source_latest_open_close[[close for close in source_latest_open_close.columns.tolist() if "close" in close]]

feats = []


t1_start = process_time()


mom1d = closes.swifter.apply(featGen.momentum, axis=0, args=(1,'D')).fillna(method='ffill').add_prefix('mom1d_')
mom5d = closes.swifter.apply(featGen.momentum, axis=0, args=(5,'D')).fillna(method='ffill').add_prefix('mom5d_')
mom10d = closes.swifter.apply(featGen.momentum, axis=0, args=(10,'D')).fillna(method='ffill').add_prefix('mom10d_')

mom5h = closes.swifter.apply(featGen.momentum, axis=0, args=(5,'H')).fillna(method='ffill').add_prefix('mom5h_')
mom1h = closes.swifter.apply(featGen.momentum, axis=0, args=(1,'H')).fillna(method='ffill').add_prefix('mom1h_')
mom10h = closes.swifter.apply(featGen.momentum, axis=0, args=(10,'H')).fillna(method='ffill').add_prefix('mom10h_')

mom30min = closes.swifter.apply(featGen.momentum, axis=0, args=(30,'min')).fillna(method='ffill').add_prefix('mom30min_')
mom15min = closes.swifter.apply(featGen.momentum, axis=0, args=(15,'min')).fillna(method='ffill').add_prefix('mom15min_')

chmom1d = mom1d.diff(periods=1, axis=0).add_prefix('chmom1d_')
chmom5d = mom5d.diff(periods=1, axis=0).add_prefix('chmom5d_')
chmom10d = mom10d.diff(periods=1, axis=0).add_prefix('chmom10d_')

chmom5h = mom5h.diff(periods=1, axis=0).add_prefix('chmom5h_')
chmom1h = mom1h.diff(periods=1, axis=0).add_prefix('chmom1h_')
chmom10h= mom10h.diff(periods=1, axis=0).add_prefix('chmom10h_')

chmom30min= mom30min.diff(periods=1, axis=0).add_prefix('chmom30min_')
chmom15min= mom15min.diff(periods=1, axis=0).add_prefix('chmom15min_')


## TODO individual currency mom
ind_currencies = set(chain.from_iterable([[x[:3], x[3:]] for x in tickers]))
li_indmom = []
for ind_currency in ind_currencies:
    mom10d_ind_pairs = mom10d[[col for col in mom10d.columns.tolist() if ind_currency in col]]
    mom5d_ind_pairs = mom5d[[col for col in mom5d.columns.tolist() if ind_currency in col]]

    mom10d_ind_mom = mom10d_ind_pairs.mean(axis = 1, skipna = True).rename("indmom10d_" + ind_currency) ## assume equal-weighting
    mom5d_ind_mom = mom5d_ind_pairs.mean(axis = 1, skipna = True).rename("indmom5d_" + ind_currency) ## assume equal-weighting
    li_indmom.extend([mom10d_ind_mom, mom5d_ind_mom])

indmom = reduce(lambda X,x: pd.merge_asof(X.sort_index().fillna(method='ffill'), x.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms')), li_indmom)

## TODO think of a way to incorporate multi indexed feats
# feats.extend([indmom])


# print(type(closes['dayofweek'].to_frame()))
# print([type(x) for x in feats])
feats_df = reduce(lambda X,x: pd.merge_asof(X.sort_index().fillna(method='ffill'), x.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms')), feats)

t1_stop = process_time()
print("Elapsed time during swifter in seconds:",t1_stop-t1_start)

# feats_df.to_pickle("data/feats_df.pkl")
# feats_df.to_csv("data/feats_df.csv")

pred_ret_5min = closes.swifter.apply(featGen.ret, n=5).shift(-5).fillna(method='ffill').add_prefix('5min_pred_ret_')

for ticker in tickers:
    X_feats = feats_df[[col for col in feats_df.columns.tolist() if ticker in col or (col.startswith("indmom") and (ticker[:3] in col or ticker[3:] in col))]]
    X_feats = X_feats.rename(str.replace(ticker,''), axis='columns')
    y = pred_ret_5min[['5min_pred_ret_' + ticker]]
    ticker_Xy = pd.merge_asof(X_feats.sort_index(), y.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms'))
    ticker_Xy['ticker'] = ticker
