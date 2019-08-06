import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.ticker as matplotticker
from tools import clean_weekends
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

"""
Here we investigate if there's any pattern from Friday 17:00 to SUnday 19:00

"""
open_closes = pd.read_pickle("data/source_latest.pkl")
open_closes = open_closes.sort_index()
open_closes['missing'] = False
#print(open_closes['2019-07-26 16:50:00'::])

start=open_closes.index.min()
end=open_closes.index.max()
full_index = pd.DataFrame(pd.date_range(start=start, end=end, freq='2T'), columns=['date'])

full_index.index = pd.to_datetime(full_index['date'], dayfirst=True)

#print(full_index)

full_open_closes = open_closes.join(full_index, how='right')
full_open_closes['missing'] = full_open_closes['missing'].fillna(True)

#print(full_open_closes['2019-07-26 16:50:00'::]['missing'])

#print(full_open_closes['2019-07-26 17:00:00'::])

missing_intervals = full_open_closes[full_open_closes['missing']]

missing_days = full_open_closes[full_open_closes['missing'] ==True].index.day

print(Counter(missing_days))

''' we want time series of 26th(Friday) close to 28th (Sunday) open and 2nd(Friday) open to 4th (Sunday) close
    We will take the +/- 1 hr to spot for patterns

'''

''' first check 30mins interval, see if returns show pattern'''
dt='30T'
first_gap = full_open_closes['2019-07-26 15:00:00':'2019-07-28 21:00:00'].resample(dt).last()
second_gap = full_open_closes['2019-08-02 15:00:00':'2019-08-04 21:00:00'].resample(dt).last()

#print(np.log(first_gap.dropna().drop(columns=['date', 'missing'])).diff().dropna())

ret_first_gap= np.log(first_gap.dropna().drop(columns=['date', 'missing'])).diff().dropna()

print(ret_first_gap.index)
## to exclude weekend gaps in graph

N, ind, date = clean_weekends.get_Nind(ret_first_gap)
def format_date(x, pos=None):
    thisind = np.clip(int(x + 0.5), 0, N - 1)
    return date[thisind].strftime('%Y-%m-%d')

f, ax = plt.subplots(figsize=(50,20))
#ax.plot(ret_first_gap)
ax.plot(ind, ret_first_gap[(-0.004 < ret_first_gap ) & ( ret_first_gap<0.005)])
#ax.axvline(x='2019-07-26 16:50:00')
#ax.axvline(x='2019-07-28 19:00:00')
ax.xaxis.set_major_formatter(matplotticker.FuncFormatter(format_date))
#ax.legend()
plt.show()

#print(np.log(first_gap).diff())
#
#print(np.log(first_gap))

# print(missing_intervals)