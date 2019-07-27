
import sys 
import pandas as pd 
import functools

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
from datetime import date


ticker = "USDTRY"

'''supported values are '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly' '''

interval = "1min"
today = date.today()
key='NB6G0K9K27IGEWXW'


ts = TimeSeries(key=key, output_format='pandas')

'''intraday data fetch'''
price, meta_data = ts.get_intraday(symbol=ticker,interval=interval, outputsize='full')

print(price)
'''daily data fetch'''
#price, meta_data = ts.get_daily(symbol=ticker, outputsize='full')


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



