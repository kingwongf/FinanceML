import pandas as pd
import numpy as np
from tools import featGen
import pickle
from functools import reduce
#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_colwidth', -1)  # or 199
#pd.set_option('display.max_rows', None)  # or 1000

open_closes = pd.read_pickle("data/source_latest.pkl")
open_closes.index = pd.to_datetime(open_closes.index, dayfirst=True)
open_closes = open_closes.sort_index()

tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

## TODO feature Day of the Week
open_closes['day_of_week'] = open_closes.index.dayofweek
day_of_week = open_closes['day_of_week'].to_frame()


## TODO feature Cointegrated Ratios
ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)

## TODO feature fractional differentiated closes
ffd_featGen = pd.read_pickle("data/generalized_LSTM/ffd_featGen_0.0001.pkl")
ffd_featGen.index = pd.to_datetime(ffd_featGen.index, dayfirst=True)
ffd_featGen.to_csv("data/test_ffd_feat.csv")

## TODO feature closes
closes = open_closes.filter(regex='close')

## TODO feature technical indicators


RSI = closes.apply(featGen.RSI)
RSI.columns = ["RSI " + ticker + "4. close " for ticker in tickers]

stochRSI_K = closes.apply(featGen.stochRSI_K)
stochRSI_K.columns = ["Stoch RSI K " + ticker + "4. close " for ticker in tickers]

stochRSI_D = closes.apply(featGen.stochRSI_D)
stochRSI_D.columns = ["Stoch RSI D " + ticker + "4. close " for ticker in tickers]

EMA_5 = closes.apply(featGen.ema,args=(5,))
EMA_5.columns = ["EMA 5 " + ticker + "4. close " for ticker in tickers]
EMA_10 = closes.apply(featGen.ema,args=(10,))
EMA_10.columns = ["EMA 10 " + ticker + "4. close " for ticker in tickers]
EMA_200 = closes.apply(featGen.ema,args=(200,))
EMA_200.columns = ["EMA 200 " + ticker + "4. close " for ticker in tickers]

MACD = closes.apply(featGen.MACD)
MACD.columns = ["MACD " + ticker + "4. close " for ticker in tickers]

ta_list = [RSI, stochRSI_K, stochRSI_D, EMA_5, EMA_10, EMA_200, MACD]

#print([ta.columns for ta in ta_list])

TA = pd.DataFrame([], index=closes.index)
TA = reduce(lambda TA,ta:pd.merge_asof(TA.dropna().sort_index(), ta.sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms')),ta_list)

## TODO gen X

xs = [ffd_featGen, ratios,day_of_week, TA, closes]
X = pd.DataFrame([], index=open_closes.index)
'''
for i,x in enumerate(xs):
    print(i)
    if i==0:
        X = x
    else:
        X = pd.merge_asof(X.dropna().sort_index(),x.sort_index(), left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms'))


'''
#for x in xs:
#    X = pd.merge_asof(X.dropna().sort_index(), x.sort_index(), left_index=True, right_index=True,
#                  direction='forward', tolerance=pd.Timedelta('2ms'))

X = reduce(lambda X,x:pd.merge_asof(X.dropna().sort_index(), x.sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms')),xs)

print(X.shape)

X.to_pickle("data/generalized_LSTM/feat_generalized_LSTM.pkl")
#print(X.sort_index())
#print("non Nan count /n", X.size)
#print("Nan count /n", X.isna().sum())
