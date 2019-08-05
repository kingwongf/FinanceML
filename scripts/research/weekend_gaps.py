import pandas as pd


open_closes = pd.read_pickle("data/source_latest.pkl")

print(open_closes.sort_index())