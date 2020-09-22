import time
import numpy as np
import pandas as pd
from hrvanalysis import get_time_domain_features

def read_data():
	with open("data.txt", "r") as f:
		data = eval(f.readline())
		labels = eval(f.readline())
		labels_reversed = eval(f.readline())
	return data, labels, labels_reversed

start_time = time.time()
data, labels, labels_reversed = read_data()
case = 1

if case == 1:
	lists = {0: ["N"], 1: ["A"]}
elif case == 2:
	lists = {0: ["A"], 1: ["N", "O", "~"]}

df = None

for key in lists:
	for category in lists[key]:
		for id_ in labels_reversed[category]:
			if len(data[id_]) > 5:
				features = get_time_domain_features(data[id_])
				if df is None:
					cols = sorted(features) + ["class"]
					df = pd.DataFrame(columns=cols)
				features["class"] = key
				df = df.append(features, ignore_index=True)

df.to_csv("case_{}.csv".format(case))

print (time.time() - start_time)