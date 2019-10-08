from datetime import date
import os
from functools import reduce
import swifter  ## needed for py 3.7 import settings
import pandas as pd
import swifter
import matplotlib.pyplot as plt
from tools import step2_feat_swifter_tools

# pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
# pd.set_option('display.max_colwidth', -1)  # or 199


''' merge dataframes of different dates to have one large dataframe '''


tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

tickers = ['EURGBP', 'EURUSD']


fx_loc = "data/fx_prices/2019-10-08"
today = date.today()
date_parser = pd.to_datetime


source_latest_open_close = pd.read_pickle("data/open_closes_source_latest_%s.pkl"%today).sort_index()
closes = source_latest_open_close[[close for close in source_latest_open_close.columns.tolist() if "close" in close]]


def read_df_format_datetime(file, root):
    df = pd.read_csv(root + "/" + file, date_parser=date_parser)
    df.index = pd.to_datetime(df['date'], dayfirst=True)
    return df

feat_df_li =[]

for ticker in tickers:
    li_full_hist_ticker =[]
    for root, dirs, files in os.walk(fx_loc):
        # print(root)
        if '.DS_Store' not in files and len(files) != 0:
            li_full_hist_ticker.extend([read_df_format_datetime(file, root) for file in files
                                    if file.endswith(".csv") and ticker in file])
            # print(li_full_hist_ticker)
    # [print(type(x)) for x in li_full_hist_ticker]
    df_full_hist_ticker = reduce(lambda X, x: X.sort_index().append(x.sort_index()), li_full_hist_ticker)
    # print(df_full_hist_ticker)
    # print(type(df_full_hist_ticker))
    df_full_hist_ticker = df_full_hist_ticker.loc[~df_full_hist_ticker.index.duplicated(keep='first')]

    ## last element as pred freq in min
    df_ticker_feat = step2_feat_swifter_tools.feat_ticker(df_full_hist_ticker[ticker +" 4. close"].to_frame(),
                                                          closes, ticker, ticker +" 4. close", 1)

    # print(df_ticker_feat[['close','target', 'prev_ret', 'ret']])

    '''
    fig, axs = plt.subplots(3, 1, figsize=(30, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs[0].plot(df_ticker_feat['RSI'].reset_index(drop=True)[100:250])
    axs[1].plot(df_ticker_feat['close'].reset_index(drop=True)[100:250])
    axs[2].plot(df_ticker_feat['target'].reset_index(drop=True)[100:250])

    axs[0].grid(which='minor', alpha=0.2)
    axs[1].grid(which='minor', alpha=0.2)
    axs[2].grid(which='minor', alpha=0.2)
    # plt.show()
    plt.close()
    '''

    # print(df_ticker_feat[df_ticker_feat.isnull().any(axis=1)])
    # print(df_ticker_feat.dropna(axis=0))
    # df[df.isnull().any(axis=1)]
    df_ticker_feat = df_ticker_feat[:-1]
    feat_df_li.append(df_ticker_feat.fillna(method='ffill').reset_index(drop=True))

# print(feat_df_li)
Xy = reduce(lambda X,x: X.reset_index(drop=True).append(x), feat_df_li)

# Xy.to_csv('data/fx_empirical_asset_pricing_via_ml/Xy.csv', index=False)
# Xy.to_pickle('data/fx_empirical_asset_pricing_via_ml/Xy.pkl')



