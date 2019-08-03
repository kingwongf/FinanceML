
### Approach 3, using a MA crossover strategy to decide the side of the bet then a Neural Net to decide to trade or not

closes['MA_5'] = featGen.ema(pairs_trading_ratio, 5) ## 5*dt, 30mins
closes['MA_10'] = featGen.ema(pairs_trading_ratio, 10)
closes['MA_200'] = featGen.ema(pairs_trading_ratio, 200)

def get_up_cross(fast, slow):
    crit1 = fast.shift(1) < slow.shift(1) ## before
    crit2 = fast > slow
    return fast[(crit1) & (crit2)]

def get_down_cross(fast, slow):
    crit1 = fast.shift(1) > slow.shift(1)
    crit2 = fast < slow
    return fast[(crit1) & (crit2)]

up = get_up_cross(closes['MA_10'], closes['MA_200'])
down = get_down_cross(closes['MA_10'], closes['MA_200'])

f, ax = plt.subplots(figsize=(11,8))

closes['ratio'].plot(ax=ax, alpha=.5)
closes['MA_10'].plot(ax=ax, label='MA 10')
closes['MA_200'].plot(ax=ax, label='MA 200')
up.plot(ax=ax,ls='',marker='^', markersize=7,
                     alpha=0.75, label='upcross', color='g')
down.plot(ax=ax,ls='',marker='v', markersize=7,
                       alpha=0.75, label='downcross', color='r')

ax.legend()
plt.show()
'''
closes['EURNZD_pos'].plot(ax=axs[0], ls='',marker='^', markersize=7,
                     alpha=0.75, label='profit taking', color='g')

'''



'''
print(closes[['ratio','long', 'short']])

closes.loc[closes['long'] == 1.0, ''] = closes['EURNZD 4. close']
closes.loc[closes['short'] == -1.0, 'bin_neg'] = Xy['4. close']
'''