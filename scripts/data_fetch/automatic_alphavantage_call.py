import sys
from datetime import date
import time
import os

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']


## TODO rename all dir to explicit because bash/ terminal doesn't sepcify local dir
keys_file = open("/Users/kingf.wong/Development/FinanceML/resources/keys.txt", "r")

for lines in keys_file:
    keys = lines.split(",")

key = keys[2]


'''supported values are '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly' '''



interval = "1min"
today = date.today()
today_dir = "/Users/kingf.wong/Development/FinanceML/data/" + str(today)
os.mkdir(today_dir)

for i,ticker in enumerate(tickers):
    if i%5==0 and i!=0:
        print("1 min wait triggered")
        time.sleep(60)
    ts = TimeSeries(key=key, output_format='pandas')
    price, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize='full')
    price.rename(columns=lambda x: ticker + " " + x, inplace=True)
    price.to_csv(today_dir + "/" + interval + '_price_' + ticker + "_" + str(today) + '.csv')
    print("finished " + ticker + " " + str(today))



