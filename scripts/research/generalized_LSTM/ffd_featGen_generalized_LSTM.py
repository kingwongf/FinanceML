from tools import FFD
import multiprocessing
from multiprocessing import Process, Pool
import pandas as pd
from itertools import product
import pickle
import matplotlib.pyplot as plt


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

#tickers = ['EURGBP']

closes = open_closes.filter(regex='close')
#closes.to_csv("data/closes_source_latest.csv")

li_closes = [closes[ticker + " 4. close"] for ticker in tickers]
ds = [ x/100 for x in range(0,100)]
import time

start = time.time()

opt_ds = [FFD.test_get_optimal_ffd(close, ds) for close in li_closes]

tau=1e-4

ffd_feat = pd.DataFrame([], index=closes.index)

for i,ticker in enumerate(tickers):
    closes[ticker + " 4. close"].index = pd.to_datetime(closes[ticker + " 4. close"].index, dayfirst=True)
    if i==0:
        #print(" opt d ", opt_ds)
        ffd_feat = FFD.fracDiff_FFD(closes[ticker + " 4. close"].to_frame(), opt_ds[i],tau)
        #print("length of time series ", len(closes[ticker + " 4. close"]))
        #print("length of frac diff ", len(ffd_feat))
    else:
        ffd_feat = pd.merge_asof(ffd_feat.sort_index(), FFD.fracDiff_FFD(closes[ticker + " 4. close"].to_frame(), opt_ds[i],tau).sort_index(),
                                 left_index=True, right_index=True,
                                 direction='forward', tolerance=pd.Timedelta('2ms')
                                 )
ffd_feat = ffd_feat.add_prefix("frac_diff ")
ffd_feat.to_pickle("data/generalized_LSTM/ffd_featGen_" +str(tau)+".pkl")
ffd_feat.to_csv("data/generalized_LSTM/ffd_featGen_" +str(tau)+".csv")
end = time.time()
print(end - start)

#f, ax = plt.subplots(figsize=(11,5))
#ax2 = ax.twinx()
#ax2.plot(ffd_feat, 'g')
#ax.plot(closes['EURGBP 4. close'])
#plt.show()
#pickle.dump(res, open("data/generalized_LSTM/list_comp_d_ffd_featGen.pkl", 'wb'))

#tup = [(open_closes['AUDCAD 4. close'], x/100) for x in range(0,100)]

#print(tup)

'''

## took 1hr 47min and don't know how to write
with Pool(multiprocessing.cpu_count()) as p:
    res = p.starmap(FFD.min_get_optimal_ffd, product(li_closes, ds))
pickle.dump(res, open("data/generalized_LSTM/d_ffd_featGen.pkl", 'wb'))
'''




