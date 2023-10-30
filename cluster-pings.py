import pandas as pd
import numpy as np
import seaborn as sns
import kmeans1d
df = pd.read_csv('mainnet.csv', delimiter='\t', header=None)

df_times = df[df[3] != "TIMEOUT"]
vec = df_times.iloc[:,3]
pings = np.array(vec, dtype=float)
k = 4
# https://pypi.org/project/kmeans1d/

clusters, centroids = kmeans1d.cluster(pings, k)


df = pd.DataFrame(data=list(zip(pings,clusters)))
# plot = df.plot.scatter(x=0,y=1)
plot = sns.scatterplot(data=df, x=0,y=1)
plot.get_figure().savefig("out.png")

