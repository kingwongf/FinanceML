
import functools
import sys
from datetime import date

import pandas as pd

from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

tickers = [['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD']
            ,['CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF']
            ,['EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK']]

#
#            , "EURAUD", "EURNZD", "USDSEK"
#            , "EURUSD", "USDCAD", "EURGBP"
#            , "GBPUSD", "CHFJPY", "EURNOK"
#            . "AUDCAD"]


keys = ['NB6G0K9K27IGEWXW', 'K3NSH7AF0NABI13X', 'FUMX5ZM974HWS3X1']
interval = "1min"
today = date.today()
#ticker = "USDTRY"


for i, ticker in enumerate(tickers):
    ts = TimeSeries(key=keys[i], output_format='pandas')
    for j in range(len(ticker[i]) - 1 ):
        print(keys[i])
        price, meta_data = ts.get_intraday(symbol=ticker[j],interval=interval, outputsize='full')
        price.rename(columns=lambda x: ticker[j] + " " + x, inplace=True)
        price.to_csv(interval + '_price_' + ticker[j] + "_" + str(today) + '.csv') 



'''supported values are '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', 'monthly' '''



#key='NB6G0K9K27IGEWXW'
#key='K3NSH7AF0NABI13X'


#ts = TimeSeries(key=key, output_format='pandas')

'''intraday data fetch'''
#price, meta_data = ts.get_intraday(symbol=ticker,interval=interval, outputsize='full')

#print(price)
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
#price.to_csv(interval + '_price_' + ticker + "_" + str(today) + '.csv')
