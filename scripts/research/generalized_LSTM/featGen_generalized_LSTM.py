import pandas as pd
import numpy as np
import pickle
from functools import reduce
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
pd.set_option('display.max_rows', None)  # or 1000

open_closes = pd.read_pickle("data/source_latest.pkl")
open_closes.index = pd.to_datetime(open_closes.index, dayfirst=True)
open_closes = open_closes.sort_index()


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

## TODO gen X

xs = [ffd_featGen, ratios,day_of_week]
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
