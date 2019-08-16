import pickle
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

open_closes = pd.read_pickle("data/source_latest.pkl")
open_closes.index = pd.to_datetime(open_closes.index, dayfirst=True)


x1_trgt = pickle.load(open("data/generalized_LSTM/labelling/x1_trgt.pkl", 'rb'))
x2_trgt= pickle.load(open("data/generalized_LSTM/labelling/x2_trgt.pkl", 'rb'))
x1_events= pickle.load(open("data/generalized_LSTM/labelling/x1_events.pkl", 'rb'))
x2_events = pickle.load(open("data/generalized_LSTM/labelling/x2_events.pkl", 'rb'))
x1_labels = pickle.load(open("data/generalized_LSTM/labelling/x1_labels.pkl", 'rb'))
x2_labels = pickle.load(open("data/generalized_LSTM/labelling/x2_labels.pkl", 'rb'))

ratios = pd.read_pickle("data/cointegrated_source_latest.pkl")
ratios.index = pd.to_datetime(ratios.index, dayfirst=True)

## TODO passe names of labelled closes
x1_closes_ratios = [close[2:17] for close in ratios]
x2_closes_ratios = [close[21:36] for close in ratios]

## TODO: plot to check if they are correctly labelled

#y_1 = pickle.load(open("data/generalized_LSTM/x1_labels.pkl", 'rb'))


x1_labelPlot = [pd.DataFrame(open_closes[close], index=open_closes[close].index) for close in open_closes[x1_closes_ratios]]


#print(x1_labelPlot.columns)
for i,label in enumerate(x1_labels):

    labelPlot = pd.merge_asof(x1_labelPlot[i].sort_index(), label,
                                 left_index=True, right_index=True, direction='forward'
                                 , tolerance=pd.Timedelta('2ms'))
    labelPlot.sort_index().to_csv("data/generalized_LSTM/labelling/investigation/label.csv")
    x1_trgt[i].sort_index().to_csv("data/generalized_LSTM/labelling/investigation/trgt.csv")
    x1_events[i].sort_index().to_csv("data/generalized_LSTM/labelling/investigation/events.csv")
    labelPlot.loc[labelPlot['bin'] == 1.0, 'bin_pos'] = labelPlot[x1_closes_ratios[i]]
    labelPlot.loc[labelPlot['bin'] == -1.0, 'bin_neg'] = labelPlot[x1_closes_ratios[i]]

    f, ax = plt.subplots(figsize=(11, 5))
    labelPlot[x1_closes_ratios[i]].plot(ax=ax, alpha=.5, label='close')
    labelPlot['bin_pos'].plot(ax=ax, ls='', marker='^', markersize=7,
                              alpha=0.75, label='buy', color='g')
    labelPlot['bin_neg'].plot(ax=ax, ls='', marker='v', markersize=7,
                              alpha=0.75, label='sell', color='r')

    ax.legend()
    plt.title("%s min max holding period long and short signals for EURNZD" % str(5))
    # plt.savefig("resources/%s min max holding period long and short signals for EURNZD"%(maxHold*int(dt[:-1])) + readin_date )
    plt.show()
    break

