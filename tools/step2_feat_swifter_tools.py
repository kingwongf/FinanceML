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



## TODO momentum and change of momentun


def feat_ticker(close_df, closes, ticker, ticker_close):

    ''' close_df refers to one ticker/ pair dataframe with only one columnm, close_df[ticker_close] '''

    ## TODO mom and chmom
    close_df['mom1d'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(1, 'D')).fillna(method='ffill')
    close_df['mom5d'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(5, 'D')).fillna(method='ffill')
    close_df['mom10d'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(10, 'D')).fillna(method='ffill')

    close_df['mom5h'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(5, 'H')).fillna(method='ffill')
    close_df['mom1h'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(1, 'H')).fillna(method='ffill')
    close_df['mom10h'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(10, 'H')).fillna(method='ffill')

    close_df['mom30min'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(30, 'min')).fillna(method='ffill')
    close_df['mom15min'] = close_df[ticker_close].swifter.apply(featGen.momentum, axis=0, args=(15, 'min')).fillna(method='ffill')

    close_df['chmom1d'] = close_df.mom1d.diff(periods=1, axis=0)
    close_df['chmom5d'] = close_df.mom5d.diff(periods=1, axis=0)
    close_df['chmom10d'] = close_df.mom10d.diff(periods=1, axis=0)

    close_df['chmom5h'] = close_df.mom5h.diff(periods=1, axis=0)
    close_df['chmom1h'] = close_df.mom1h.diff(periods=1, axis=0)
    close_df['chmom10h'] = close_df.mom10h.diff(periods=1, axis=0)

    close_df['chmom30min'] = close_df.mom30min.diff(periods=1, axis=0)
    close_df['chmom15min'] = close_df.mom15min.diff(periods=1, axis=0)

    ## TODO ind mom
    # ind_currencies = set(chain.from_iterable([[x[:3], x[3:]] for x in tickers]))
    ind_currencies_top, ind_currencies_bottom = closes[[col for col in closes.columns.tolist() if col.startswith(ticker[:3])]], \
                                                closes[[col for col in closes.columns.tolist() if
                                                        ticker[:3] in col.name[:3]]]

    # print(ind_currencies_top.columns, ind_currencies_bottom.columns)

    ind_mom = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(15, 'min'))\
        .fillna(method='ffill').mean(axis=1, skipna=True).rename('top_ind_mom15min')

    ind_mom['top_ind_mom30min'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(30, 'min')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom1h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(1, 'H')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom5h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(5, 'H')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom10h'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(10, 'H')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom1d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(1, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom5d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(5, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['top_ind_mom10d'] = ind_currencies_top.swifter.apply(featGen.momentum, axis=0, args=(10, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom15min'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(15, 'min'))\
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom30min'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(30, 'min')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom1h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(1, 'H'))\
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom5h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(5, 'H')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom10h'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(10, 'H')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom1d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(1, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom5d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(5, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    ind_mom['bottom_ind_mom10d'] = ind_currencies_bottom.swifter.apply(featGen.momentum, axis=0, args=(10, 'D')) \
        .fillna(method='ffill').mean(axis=1, skipna=True)

    close_df = pd.merge_asof(close_df.sort_index(), indmom.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms'))

    ## TODO return vol

    close_df['retvol1d'] = close_df[ticker_close].swifter.apply(featGen.retvol, axis=0, args=('1d',)).fillna(
        method='ffill')
    close_df['retvol5d'] = close_df[ticker_close].swifter.apply(featGen.retvol, axis=0, args=('5d',)).fillna(
        method='ffill')
    close_df['retvol10d'] = close_df[ticker_close].swifter.apply(featGen.retvol, axis=0, args=('10d',)).fillna(
        method='ffill')

    close_df['retvol30min'] = close_df[ticker_close].swifter.apply(featGen.retvol, axis=0, args=('30min',)).fillna(method='ffill')
    close_df['retvol15min'] = close_df[ticker_close].swifter.apply(featGen.retvol, axis=0, args=('15min',)).fillna(method='ffill')

    ## TODO maxret

    close_df['maxret1d'] = close_df[ticker_close].swifter.apply(featGen.maxret, axis=0, args=('1d',)).fillna(method='ffill')
    close_df['maxret5d'] = close_df[ticker_close].swifter.apply(featGen.maxret, axis=0, args=('5d',)).fillna(method='ffill')

    ## TODO datetime feat

    close_df['dayofweek'] = close_df.index.dayofweek

    ## TODO add ticker

    close_df['ticker'] = ticker_close

    ## TODO add label

    close_df['1h_pred_ret'] = closes.swifter.apply(featGen.ret, n=60).shift(-60).fillna(method='ffill')

    return close_df


'''
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


# print(mom1d['2019-09-19 06:38:00':'2019-09-19 06:47:00'], non_parall_mom1d['2019-09-19 06:38:00':'2019-09-19 06:47:00'])

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

def set_multiColIndex(df, feat):
    df.columns = pd.MultiIndex.from_product([df.columns, [feat]])
    return df

# li_mom_str =['mom1d', 'mom5d', 'mom10d', 'mom5h', 'mom1h', 'mom10h', 'mom30min', 'mom15min', 'chmom1d',
#             'chmom5d', 'chmom10d', 'chmom5h', 'chmom1h', 'chmom10h', 'chmom30min', 'chmom15min']


mom1d = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(1,'D')).fillna(method='ffill'), 'mom1d')
mom5d = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(5,'D')).fillna(method='ffill'),'mom5d')
mom10d = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(10,'D')).fillna(method='ffill'),'mom10d')

mom5h = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(5,'H')).fillna(method='ffill'),'mom5h')
mom1h = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(1,'H')).fillna(method='ffill'),'mom1h')
mom10h = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(10,'H')).fillna(method='ffill'),'mom10h')

mom30min = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(30,'min')).fillna(method='ffill'),'mom30min')
mom15min = set_multiColIndex(closes.swifter.apply(featGen.momentum, axis=0, args=(15,'min')).fillna(method='ffill'),'mom15min')

chmom1d = set_multiColIndex(mom1d.diff(periods=1, axis=0),'chmom1d')
chmom5d = set_multiColIndex(mom5d.diff(periods=1, axis=0),'chmom5d')
chmom10d = set_multiColIndex(mom10d.diff(periods=1, axis=0),'chmom10d')

chmom5h = set_multiColIndex(mom5h.diff(periods=1, axis=0),'chmom5h')
chmom1h = set_multiColIndex(mom1h.diff(periods=1, axis=0),'chmom1h')
chmom10h= set_multiColIndex(mom10h.diff(periods=1, axis=0),'chmom10h')

chmom30min= set_multiColIndex(mom30min.diff(periods=1, axis=0),'chmom30min')
chmom15min= set_multiColIndex(mom15min.diff(periods=1, axis=0),'chmom15min')


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
'''