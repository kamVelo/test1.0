import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import where
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import  pyplot
symbol = "aapl"
dset = pd.read_csv(os.path.join(symbol, "dset.csv")).iloc[:, 1:].iloc[::-1]
rsi  = pd.read_csv(os.path.join(symbol, "rsi.csv")).iloc[:, 1:].iloc[::-1]
dset.index = pd.RangeIndex(start=0, stop=len(dset), step=1)


length = min([len(dset), len(rsi)])
dset = dset[:length]
rsi = rsi[:length]
sc_y = StandardScaler()
sc_x = StandardScaler()
rsi = sc_y.fit_transform(rsi)
closes = dset["close"]
closes = sc_x.fit_transform(closes.values.reshape(-1,1))
x= pd.DataFrame(rsi,columns=["rsi"])
x["closes"] = closes;
model = KMeans(init="random", n_clusters=5, n_init=3, max_iter=300, random_state=42)
model.fit(x)
x["rsi"] = sc_y.inverse_transform(x["rsi"])
x["closes"] = sc_x.inverse_transform(x["closes"])
x["prediction"] = model.labels_

blue = [row for index, row in x.iterrows() if row["prediction"] == 0]
red = [row for index, row in x.iterrows() if row["prediction"] == 1]
green = [row for index, row in x.iterrows() if row["prediction"] == 2]
black = [row for index, row in x.iterrows() if row["prediction"] == 3]
yellow = [row for index, row in x.iterrows() if row["prediction"] == 4]


blue,red , green, black, yellow = np.array(blue), np.array(red), np.array(green), np.array(black), np.array(yellow)

plt.scatter(blue[:,1],blue[:,0], c="blue")
plt.scatter(red[:,1],red[:,0], c="red")
plt.scatter(green[:,1],green[:,0], c="green")
plt.scatter(black[:,1],black[:,0], c="black")
plt.scatter(yellow[:,1],yellow[:,0], c="yellow")
plt.show()




