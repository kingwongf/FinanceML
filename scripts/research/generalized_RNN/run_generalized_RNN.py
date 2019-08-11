import pandas as pd
import numpy as np
import pickle

X = pd.read_pickle("data/generalized_RNN/feat_generalized_RNN.pkl")

y_1 = pickle.load(open("data/generalized_RNN/x1_labels.pkl", 'rb'))

#print(y_1)

ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
#print(ratios[].columns)
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)
y1_closes_names = [close[2:17] for close in ratios]

for i, y in enumerate(y_1):
    print(i)
    print(y1_closes_names[i])
    print(y.count())

#print(x1_closes_ratios)

### TODO:  test with first fx, 'EURTRY 4. close'
###        in the x1_closes_ratios as it has most labels (2951)



Xy = pd.merge_asof(X.dropna().sort_index(), y_1[39]['bin'].sort_index(), left_index=True, right_index=True,
                  direction='forward', tolerance=pd.Timedelta('2ms'))
#print(Xy.index.max, Xy.index.min)
#Xy.to_csv("data/generalized_RNN/test_Xy_RNN.csv")
print(Xy.count())