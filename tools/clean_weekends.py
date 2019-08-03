import pandas as pd

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


tickers = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD'
            ,'CADCHF', 'CADJPY', 'EURAUD', 'EURCAD', 'EURCHF'
            ,'EURGBP', 'EURJPY', 'EURNOK', 'EURNZD', 'EURSEK'
            ,'EURTRY', 'EURUSD', 'GBPAUD', 'GBPCAD', 'GBPCHF'
            ,'GBPJPY', 'GBPNZD', 'GBPUSD', 'NZDCAD', 'NZDCHF'
            ,'NZDJPY', 'NZDUSD', 'TRYJPY', 'USDCAD', 'USDCHF'
            ,'USDCNH', 'USDJPY', 'USDMXN', 'USDNOK', 'USDSEK'
            ,'USDTRY', 'USDZAR', 'ZARJPY']

interval = "1min"
date = "2019-07-31"
date_dir = "data/" + date + "/"
date_parser = pd.to_datetime
#prices = [pd.read_csv("data/" + interval + '_price_' + ticker + "_" + str(today) + '.csv', date_parser=date_parser) for ticker in tickers]
prices = [pd.read_csv( date_dir + interval + '_price_' + ticker + "_" + date + '.csv', date_parser=date_parser) for ticker in tickers]



for i,ticker in enumerate(tickers):
    prices[i].index = pd.to_datetime(prices[i]['date'], dayfirst=True)
    if i==0:
        closes = prices[i][ticker + " 4. close"]
        opens = prices[i][ticker + " 1. open"]
    else:
        closes = pd.merge_asof(closes, prices[i][ticker + " 4. close"],
                    left_index=True, right_index=True,
                    direction='forward',tolerance=pd.Timedelta('2ms')).dropna()
        opens = pd.merge_asof(opens, prices[i][ticker + " 1. open"],
                               left_index=True, right_index=True,
                               direction='forward', tolerance=pd.Timedelta('2ms')).dropna()

opens_closes = pd.merge_asof(closes, opens,
                        left_index=True, right_index=True,
                        direction='forward', tolerance=pd.Timedelta('2ms')).dropna()

print(opens_closes['26/07/2019  16:49:00': '28/07/2019  19:06:00'])


