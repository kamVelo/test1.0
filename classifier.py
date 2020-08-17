import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
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
args = {
    "init":"random",
    "n_init":5,
    "max_iter":300,
    "random_state":42
}
sse = []
max_iter = 51
for k in range(1,max_iter):
    model = KMeans(n_clusters=k, **args)
    model.fit(x)
    sse.append(model.inertia_)

k1 = KneeLocator(range(1, max_iter), sse, curve="convex",direction="decreasing")
print(k1.elbow)
silCoefs = []
for k in range(2,max_iter):
    model = KMeans(n_clusters=k, **args)
    model.fit(x)
    score = silhouette_score(x, model.labels_)
    silCoefs.append(score)
    
bestSilCoef = silCoefs.index(max(silCoefs)) + 2

print("elbow: %s | Silhouette Coefficient: %s" % (k1.elbow, bestSilCoef))

