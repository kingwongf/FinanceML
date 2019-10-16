import quandl
import pickle
import time
import pandas as pd
# quandl.ApiConfig.api_key = 'j2QGEgPyPhe-dGsNNtzE'
quandl.ApiConfig.api_version = '2015-04-09'
quandl.save_key("j2QGEgPyPhe-dGsNNtzE")
print(quandl.ApiConfig.api_key)

metadata = pd.read_csv("data/futures/CHRIS_metadata.csv")
codes = metadata['code'].tolist()
print(len(codes))
counter=0
for code in codes:
	if counter%300 ==0 and counter!=0:
		time.sleep(60)
	futures = quandl.get("CHRIS/" + code)
	futures.to_csv("data/futures/" + code +".csv")
	print("finished ", code)
	counter+=1
# print("CHRIS/" + codes[2])
# futures = quandl.get("CHRIS/" + codes[2])
# futures = quandl.get("CHRIS/" + "CME_EM1")
# print(futures)
# pickle.dump(futures, open("data/fx_empirical_asset_pricing_via_ml/futures.pkl", "wb"))