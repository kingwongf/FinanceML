import numpy as np
import sklearn.covariance
from datetime import date
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


from tools import featGen
from tools import labelling_Marcos
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


open_closes = pd.read_pickle("data/source_latest.pkl")
## generate 'ratio' according to cointegration pairs
## get tEvents according to ratios

closes = open_closes.filter(regex='close')
absCorr = closes.corr().abs()
sol = (absCorr.where(np.triu(np.ones(absCorr.shape), k=1).astype(np.bool))
                 .stack()
                 .sort_values(ascending=False)).to_frame()
sol['pairs'] = sol.index
sol = sol.set_index(np.arange(len(sol.index)))

adfStats = []
for i in range(len(sol)):
    model = sm.regression.linear_model.OLS(closes[sol['pairs'][i][0]], closes[sol['pairs'][i][1]])
    results = model.fit()
    pairAdfStats = sm.tsa.stattools.adfuller(results.resid)
    adfStats.append(pairAdfStats)

sol['adfStats'] = adfStats
coIntegrate = [(abs(x[0]) > abs(x[4]['5%'])) for x in adfStats]
sol['cointegration'] = coIntegrate


cointegratedPairs = sol[coIntegrate]
cointegratedPairs = cointegratedPairs.reset_index()

ratios = pd.DataFrame([])

for i in range(len(cointegratedPairs)):
    ratios[str(cointegratedPairs['pairs'][i])] = closes[cointegratedPairs['pairs'][i][0]] / closes[cointegratedPairs['pairs'][i][1]]


ratios.to_pickle("data/cointegrated_source_latest.pkl")