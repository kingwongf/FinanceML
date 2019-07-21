import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


EURTRY = pd.read_csv("1min_price_EURTRY_2019-07-20.csv")
EURTRY.index = pd.to_datetime(EURTRY['date'])
EURTRY.rename(columns=lambda x: "EURTRY" + " " + x, inplace=True)
USDTRY = pd.read_csv("1min_price_USDTRY_2019-07-20.csv")
USDTRY.index = pd.to_datetime(USDTRY['date'])
USDTRY.rename(columns=lambda x: "USDTRY" + " " + x, inplace=True)

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

X_prime_y = X.transpose().dot(y)
beta_hat = np.linalg.inv(X.transpose().dot(X)).dot(X_prime_y)
#print(beta_hat)

#print(X.dot(beta_hat))
#print(y[0:1], X.dot(beta_hat)[0:1])
#print(X.dot(beta_hat).sub(y, fill_value=0))

e = pd.merge_asof( y, X.dot(beta_hat), left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))

e = e.rename(columns={0: "error_1", "4. close_y": "error_2"})

#print(e.columns)

#print(e["error_1"] - e["error_2"])


def analytical_OLS(X, y):
    X, y = X.dropna(), y.dropna()
    X_prime_y = X.transpose().dot(y)
    beta_hat = np.linalg.inv(X.transpose().dot(X)).dot(X_prime_y) 

    e = pd.merge_asof(y, X.dot(beta_hat), left_index=True, right_index=True, direction='forward'
                   ,tolerance=pd.Timedelta('2ms'))
    error = e.diff(axis=1)

    return beta_hat, error[0]

beta, error = analytical_OLS(X,y)    

f, ax = plt.subplots(figsize=(11,8))
error.plot(ax=ax, alpha=.5, label='error')
#plt.show()
## stats model 

reg1 = sm.OLS(endog=y, exog=X, missing='drop')
results = reg1.fit()
#print(results.summary())

''' ADF test 
delta_y_t = alpha + beta*t + theta* Y_(t-1) + summ (lambda_(lag i -1 ) * delta_y_(lag t - i +1 ) + u_t
'''

deltaYt = y.diff(axis=0)
alpha = 0
X['trend'] = range(0, len(X))
print(X['trend'])

