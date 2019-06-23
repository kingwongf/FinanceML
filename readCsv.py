import pandas as pd
import matplotlib.pyplot as plt


price = pd.read_csv('out.csv')

price.index = pd.to_datetime(price.index)

#print(df)

#df['ROC10Min'] = 

def getDailyVol(close, span0 = 100):
    # daily vol, reindexed to close
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    print(sum(df0))
    #df0 = df0[df0>0]
    #df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    

getDailyVol(price['4. close'])


""" df.plot()
plt.title('Intraday Times Series for the EURUSD (1 min)')
plt.show() """