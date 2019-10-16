
import sys 
import pandas as pd 
import functools

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
from datetime import date

from datetime import date
import time
import os

tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

# missing 'USDCNH',
# tickers =['USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
#             ,'USDTRY', 'USDZAR', 'ZARJPY']

'''supported values are '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly' '''

interval = "daily"
today = date.today()
keys_file = open("/Users/kingf.wong/Development/FinanceML/resources/keys.txt", "r")

for lines in keys_file:
    keys = lines.split(",")

key = keys[1]

today_dir = "/Users/kingf.wong/Development/FinanceML/data/fx_daily_prices/" + str(today)
# os.mkdir(today_dir)

ts = TimeSeries(key=key, output_format='pandas')

'''intraday data fetch'''
# price, meta_data = ts.get_intraday(symbol=ticker,interval=interval, outputsize='full')


'''daily data fetch'''

ts = TimeSeries(key=key, output_format='pandas')
price, meta_data = ts.get_daily(symbol=tickers[0], outputsize='full')
print(price)
'''
for i,ticker in enumerate(tickers):
    if i%5==0 and i!=0:
        print("1 min wait triggered")
        time.sleep(60)
    ts = TimeSeries(key=key, output_format='pandas')
    price, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
    price.rename(columns=lambda x: ticker + " " + x, inplace=True)
    price.to_csv(today_dir + "/" + interval + '_price_' + ticker + "_" + str(today) + '.csv')
    print("finished " + ticker + " " + str(today))
'''


#ta = TechIndicators(key=key, output_format='pandas')
#ema5, meta_ema5 = ta.get_ema(symbol=ticker,interval=interval, time_period='5')
#ema10, meta_ema10 = ta.get_ema(symbol=ticker,interval=interval, time_period='10')
#ema200, meta_ema10 = ta.get_ema(symbol=ticker,interval=interval, time_period='200')

#macd, meta_macd = ta.get_macd(symbol=ticker,interval=interval, series_type='close',
#                 fastperiod=None, slowperiod=None, signalperiod=None)
#bbBands, meta_bbands = ta.get_bbands(symbol=ticker, interval=interval, time_period=20,  series_type='close',
#                   nbdevup=None, nbdevdn=None, matype=None)
#rsi, meta_rsi = ta.get_rsi(symbol=ticker,interval=interval, time_period=20, series_type='close')
#stochrsi, meta_stochrsi = ta.get_stochrsi(symbol=ticker,interval=interval, time_period=20,
#                     series_type='close', fastkperiod=None, fastdperiod=None, fastdmatype=None)




#TAList = [price, macd, ema5, ema10, ema200, rsi, stochrsi]
#
#'''only works with daily for now'''
#df = functools.reduce(lambda left,right: pd.merge(left,right,on='date', how='left'), TAList)
#
#df.columns = ['date', '1. open', '2. high', '3. low', '4. close', '5. volume',
#       'MACD_Signal', 'MACD_Hist', 'MACD', 'EMA_5', 'EMA_10', 'EMA_200', 'RSI',
#       'FastD_StochRSI', 'FastK_StchRSI']



#df.to_csv(interval + '_price_' + ticker + "_" + str(today) +"_" + '.csv')
price.to_csv(interval + '_price_' + ticker + "_" + str(today) + '.csv')



