import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

model = KMeans(
    init="random",
    n_clusters=5,
    n_init= 3,
    max_iter=300,
    random_state=42
)

model.fit(x)
print("SSE: %s" % model.inertia_)
print("Number of clusters required to converge: %s" % model.n_iter_)

