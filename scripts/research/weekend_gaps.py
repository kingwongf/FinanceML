import pandas as pd

from collections import Counter


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

print(full_open_closes.groupby(['missing']).count())

# print(missing_intervals)