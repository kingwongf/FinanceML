import numpy as np
import sklearn.covariance
import datetime
from datetime import date
import os
from functools import reduce
import pandas as pd
from time import process_time
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
fx_loc = "data/fx_prices"
today = date.today()
date_parser = pd.to_datetime

def read_df_format_datetime(file, root):
    df = pd.read_csv(root + "/" + file, date_parser=date_parser)
    df.index = pd.to_datetime(df['date'], dayfirst=True)
    return df

for ticker in tickers:
    li_full_hist_ticker =[]
    for root, dirs, files in os.walk(fx_loc):
        # print(root)
        if '.DS_Store' not in files and len(files) != 0:
            li_full_hist_ticker.extend([read_df_format_datetime(file, root) for file in files
                                    if file.endswith(".csv") and ticker in file])

    df_full_hist_ticker = reduce(lambda X, x: X.sort_index().append(x.sort_index()), li_full_hist_ticker)
    df_full_hist_ticker = df_full_hist_ticker.loc[~df_full_hist_ticker.index.duplicated(keep='first')].sort_index().interpolate()
    print(df_full_hist_ticker.sort_index().index)




'''
def read_df_format_datetime(files, root):
    dfs = [pd.read_csv(root+ "/" + file, date_parser=date_parser) for file in files if file.endswith(".csv")]
    for df in dfs:
        df.index = pd.to_datetime(df['date'], dayfirst=True)
    dfs = [df.sort_index()[[col for col in df.columns.tolist() if "close" in col or "open" in col]].interpolate() for df in dfs]
    dfs = reduce(lambda X,x: pd.merge_asof(X.sort_index(), x.sort_index(),
                        left_index=True, right_index=True, direction='forward',tolerance=pd.Timedelta('2ms')), dfs)
    return dfs


source_latest = reduce(lambda X,x: X.sort_index().append(x.sort_index()), [read_df_format_datetime(files,root) for root, _, files in os.walk(fx_loc) if '.DS_Store' not in files if len(files)!=0])
source_latest = source_latest.loc[~source_latest.index.duplicated(keep='first')].sort_index().interpolate()

source_latest.to_pickle("data/open_closes_source_latest_%s.pkl"%today)
source_latest.to_csv("data/open_closes_source_latest_%s.csv"%today)
'''