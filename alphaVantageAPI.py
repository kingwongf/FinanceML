
import sys 
import pandas as pd 
from yahoo_finance_api2 import share 
from yahoo_finance_api2.exceptions import YahooFinanceError

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt

ts = TimeSeries(key='7KGPPA617SS5ZKZO', output_format='pandas')
price, meta_data = ts.get_intraday(symbol='EURUSD',interval='1min', outputsize='full')

ta = TechIndicators(key='7KGPPA617SS5ZKZO', output_format='pandas')
ema5, meta_ema5 = ta.get_ema(symbol='EURUSD',interval='1min', time_period='5')
ema10, meta_ema10 = ta.get_ema(symbol='EURUSD',interval='1min', time_period='10')
ema200, meta_ema10 = ta.get_ema(symbol='EURUSD',interval='1min', time_period='200')

TAList = [ema5, ema10, ema200]
TAName = ['5minEMA', '10minEMA', '200minEMA']


#df = price['4. close'].to_frame()
price.index = pd.to_datetime(price.index)


for i in range(len(TAList)):
    TAList[i].index = pd.to_datetime(TAList[i].index)
    price[TAName[i]] = TAList[i]

""" df['5minEMA'] = ema5
df['10minEMA'] = ema10
df['200minEMA'] = ema200
 """
print(df)

df.to_csv('out.csv')



#print(price['4. close'], ema5, ema10, ema200)
""" data['4. close'].plot()
plt.title('Intraday Times Series for the EURUSD (1 min)')
plt.show() """


""" my_share = share.Share('MSFT')
symbol_data = None

try: symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY, 100, share.FREQUENCY_TYPE_DAY,1)
except YahooFinanceError as e:
    print(e.message)
    sys.exit(1)

df = pd.DataFrame(symbol_data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

#df['5D_Rolliing'] = df.rolling()

print(df)

7KGPPA617SS5ZKZO """