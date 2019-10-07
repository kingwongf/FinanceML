import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
from itertools import chain
from functools import reduce
from time import process_time

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


def feat_ticker(close_df, closes, ticker, ticker_close, pred_freq):

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

