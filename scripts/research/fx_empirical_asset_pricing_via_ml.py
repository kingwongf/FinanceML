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
t1_stop = process_time()

t2_start = process_time()
non_parall_mom1d = closes.apply(featGen.momentum, axis=0, args=(1,'D')).fillna(method='ffill').add_prefix('mom1d_')
t2_stop = process_time()

print("Elapsed time during swifter in seconds:",
                                         t1_stop-t1_start)
print("Elapsed time during non parallel in seconds:",
                                         t2_stop-t2_start)

# print(mom1d['2019-09-19 06:38:00':'2019-09-19 06:47:00'], non_parall_mom1d['2019-09-19 06:38:00':'2019-09-19 06:47:00'])
'''

## TODO momentum and change of momentun
mom1d = closes.apply(featGen.momentum, axis=0, args=(1,'D')).fillna(method='ffill').add_prefix('mom1d_')
mom5d = closes.apply(featGen.momentum, axis=0, args=(5,'D')).fillna(method='ffill').add_prefix('mom5d_')
mom10d = closes.apply(featGen.momentum, axis=0, args=(10,'D')).fillna(method='ffill').add_prefix('mom10d_')

mom5h = closes.apply(featGen.momentum, axis=0, args=(5,'H')).fillna(method='ffill').add_prefix('mom5h_')
mom1h = closes.apply(featGen.momentum, axis=0, args=(1,'H')).fillna(method='ffill').add_prefix('mom1h_')
mom10h = closes.apply(featGen.momentum, axis=0, args=(10,'H')).fillna(method='ffill').add_prefix('mom10h_')

mom30min = closes.apply(featGen.momentum, axis=0, args=(30,'min')).fillna(method='ffill').add_prefix('mom30min_')
mom15min = closes.apply(featGen.momentum, axis=0, args=(15,'min')).fillna(method='ffill').add_prefix('mom15min_')

chmom1d = mom1d.diff(periods=1, axis=0).add_prefix('chmom1d_')
chmom5d = mom5d.diff(periods=1, axis=0).add_prefix('chmom5d_')
chmom10d = mom10d.diff(periods=1, axis=0).add_prefix('chmom10d_')

chmom5h = mom5h.diff(periods=1, axis=0).add_prefix('chmom5h_')
chmom1h = mom1h.diff(periods=1, axis=0).add_prefix('chmom1h_')
chmom10h= mom10h.diff(periods=1, axis=0).add_prefix('chmom10h_')

chmom30min= mom30min.diff(periods=1, axis=0).add_prefix('chmom30min_')
chmom15min= mom15min.diff(periods=1, axis=0).add_prefix('chmom15min_')

feats.extend([mom1d, mom5d, mom10d, mom5h, mom1h, mom10h, mom30min, mom15min, chmom1d,
             chmom5d, chmom10d, chmom5h, chmom1h, chmom10h, chmom30min, chmom15min])

## TODO individual currency mom
ind_currencies = set(chain.from_iterable([[x[:3], x[3:]] for x in tickers]))
li_indmom = []
for ind_currency in ind_currencies:
    mom10d_ind_pairs = mom10d[[col for col in mom10d.columns.tolist() if ind_currency in col]]
    mom5d_ind_pairs = mom5d[[col for col in mom5d.columns.tolist() if ind_currency in col]]

    mom10d_ind_mom = mom10d_ind_pairs.mean(axis = 1, skipna = True).rename("indmom10d_" + ind_currency) ## assume equal-weighting
    mom5d_ind_mom = mom5d_ind_pairs.mean(axis = 1, skipna = True).rename("indmom5d_" + ind_currency) ## assume equal-weighting
    li_indmom.extend([mom10d_ind_mom, mom5d_ind_mom])

indmom = reduce(lambda X,x: pd.merge_asof(X.sort_index().interpolate(), x.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms')), li_indmom)
feats.extend(indmom)

## TODO return vol

retvol1d = closes.apply(featGen.retvol, axis=0, args=('1d',)).fillna(method='ffill').add_prefix('retvol1d_')
retvol5d = closes.apply(featGen.retvol, axis=0, args=('5d',)).fillna(method='ffill').add_prefix('retvol5d_')
retvol10d = closes.apply(featGen.retvol, axis=0, args=('10d',)).fillna(method='ffill').add_prefix('retvol10d_')

retvol30min = closes.apply(featGen.retvol, axis=0, args=('30min',)).fillna(method='ffill').add_prefix('retvol30min_')
retvol15min = closes.apply(featGen.retvol, axis=0, args=('15min',)).fillna(method='ffill').add_prefix('retvol15min_')


feats.extend([retvol1d, retvol5d, retvol10d, retvol30min, retvol15min])

## TODO maxret

maxret1d = closes.apply(featGen.maxret, axis=0, args=('1d',)).fillna(method='ffill').add_prefix('maxret1d_')
maxret5d = closes.apply(featGen.maxret, axis=0, args=('5d',)).fillna(method='ffill').add_prefix('maxret5d_')

feats.extend([maxret1d, maxret5d])


## TODO datetime feat

closes['dayofweek'] = closes.index.dayofweek

feats.extend(closes['dayofweek'].to_frame())
'''

print(feats)