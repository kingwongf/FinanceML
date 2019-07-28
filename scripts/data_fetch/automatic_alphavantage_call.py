import sys
from datetime import date
import random
import time

import pandas as pd

from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']





keys_file = open("keys.txt", "r")

for lines in keys_file:
    keys = lines.split(",")

key = keys[0]

'''supported values are '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly' '''



interval = "1min"
today = date.today()


for i,ticker in enumerate(tickers):
    if i%5==0 and i!=0:
        print("1 min wait triggered")
        time.sleep(60)
    ts = TimeSeries(key=key, output_format='pandas')
    price, meta_data = ts.get_intraday(symbol=ticker, interval=interval, outputsize='full')
    price.rename(columns=lambda x: ticker + " " + x, inplace=True)
    price.to_csv("data/" + "2019-07-29" + "/" + interval + '_price_' + ticker + "_" + str(today) + '.csv')
    print("finished " + ticker + " " + str(today))



