import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
pd.options.display.width = 0



EURTRY = pd.read_csv("1min_price_EURTRY_2019-07-22.csv")
EURTRY.index = pd.to_datetime(EURTRY['date'])
USDTRY = pd.read_csv("1min_price_USDTRY_2019-07-22.csv")
USDTRY.index = pd.to_datetime(USDTRY['date'])

Xy = pd.merge_asof(EURTRY['EURTRY 4. close'],USDTRY[['USDTRY 4. close', 'USDTRY 1. open']],
                   left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))

Xy = Xy.dropna()


                   
#Xy.columns['X', 'y']



#print(Xy)

#var = np.linalg.inv((Xy['4. close_x'])).T


#beta_hat = (X' X)^-1 X' y
#
#

X = Xy[['USDTRY 4. close', 'USDTRY 1. open']]
y = Xy[['EURTRY 4. close']]


def analytical_OLS(olsX, olsY):
    #X, y = X.dropna(), y.dropna()
    #print(X.size(), y.size())

    X_prime_y = olsX.transpose().dot(olsY)

    beta_hat = np.linalg.inv(olsX.transpose().dot(olsX)).dot(X_prime_y)


    e = pd.merge_asof(olsY, olsX.dot(beta_hat).rename("pre_e"), left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))
#    print(e)

    error = e.diff(axis=1)

    return beta_hat, error

#beta, error = analytical_OLS(X,y)


#f, ax = plt.subplots(figsize=(11,8))
#error.plot(ax=ax, alpha=.5, label='error')
#plt.show()
## stats model 

reg1 = sm.OLS(endog=y, exog=X, missing='drop')
results = reg1.fit()
#print(results.summary())

''' ADF test 
delta_y_t = alpha + beta*t + theta* Y_(t-1) + summ (lambda_(lag i -1 ) * delta_y_(lag t - i +1 ) + u_t
'''
y['delta_yt'] = y['EURTRY 4. close'].diff()




alpha = 0
y['trend'] = range(0, len(y['EURTRY 4. close']))
y['y_lag1'] = y['EURTRY 4. close'].shift(-1)
for lag in range(len(y) - 10):
    y['delta_y_lag'+ str(lag)] = y['EURTRY 4. close'].shift(lag).diff()




y = y[['trend', 'y_lag1', 'delta_y_lag2', 'delta_y_lag3', 'delta_y_lag4', 'delta_y_lag5', 'delta_y_lag6', 'delta_yt']]
y = y.dropna()

adf_regressors, ut = analytical_OLS(y[['trend', 'y_lag1', 'delta_y_lag2', 'delta_y_lag3', 'delta_y_lag4', 'delta_y_lag5', 'delta_y_lag6']],y['delta_yt'])

print(adf_regressors, ut)


