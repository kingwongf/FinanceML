import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, tqdm_notebook
import mp
import labelling_Marcos

#print(pd.__version__)
#price = pd.read_csv("1min_price_EURTRY_2019-07-20_1min.csv")

#price.index = pd.to_datetime(price['date'])

#df = df.sort_index()
#name = 'EMA_' % span

def ema(close, span):
    ema = close.ewm(span=span,adjust=False,ignore_na=False).mean()
    return ema

def relEMA(fast_ema, slow_ema):
    return fast_ema/ slow_ema

def RSI(close, period):
    delta = close.diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    rollUp = dUp.ewm(span=period).mean()
    rollDown = dDown.abs().ewm(span=period).mean()
    rsi = rollUp/ rollDown
    RSI = 100.0 - (100.0 / (1.0 + rsi))
    return RSI

def stochRSI(close, period=14):
    rsi = RSI(close, period)
    rsiLow = rsi.rolling(period).min()
    rsiHigh = rsi.rolling(period).max()
    K = 100*(rsi - rsiLow)/ (rsiHigh - rsiLow)
    D = K.rolling(3).mean()
    return K, D

def MACD(close):
    shortEma = close.ewm(adjust=True, alpha=0.15).mean()
    longEma = close.ewm(adjust=True, alpha=0.075).mean()
    macd = shortEma - longEma
    return macd




#K, D = stochRSI(price['4. close'])

#print(MACD(price['4. close']))

#print(price[['EMA_5', 'pandas_5days_EMA']])

#print(stochRSI()