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
import zipfile
import time


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


''' merge dataframes of different dates to have one large dataframe '''


tickers = ['AUDJPY', 'AUDNZD', 'AUDUSD'
            , 'CADJPY', 'CHFJPY', 'EURCHF'
            , 'EURGBP', 'EURJPY', 'EURUSD'
            , 'GBPJPY', 'GBPUSD', 'NZDUSD'
            , 'USDCAD', 'USDCHF', 'USDJPY']

fx_loc = "data/test_truefx_tick"
today = date.today()
date_parser = pd.to_datetime

t = time.process_time()

for root, _, files in os.walk(fx_loc):
    dfs = []
    files.remove('.DS_Store')
    if len(files)!=0:
        for file in files:
            ticker = file[:6]
            if ticker in tickers:
                with zipfile.ZipFile(root + "/" + file, "r") as zip_ref:
                    zip_ref.extractall(root)
                csv_path = root + "/" + file[:-4] + ".csv"
                df = pd.read_csv(csv_path, header=None, names=['ticker', 'date', ticker + '_bid', ticker + '_ask'], low_memory=False)
                os.remove(csv_path)
                df.index = pd.to_datetime(df.date)
                df = df.drop('ticker',axis=1)
                dfs.append(df)
        dfs_ = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(),
                                            left_index=True, right_index=True, direction='forward',
                                            tolerance=pd.Timedelta('2ms')), dfs)
        dfs_ = dfs_.resample('D').first()
        dfs_.to_pickle(root + "/daily_fx.pkl")
elapsed_time = time.process_time() - t
print(elapsed_time, " s")

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