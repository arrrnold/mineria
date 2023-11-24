# probar cada uno de los metodos de linkage

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 3],
    [9, 3]
])

linkage_methods = ["single", "complete", "average", "centroid", "ward"]

fig, axes = plt.subplots(1, len(linkage_methods) + 1, figsize=(20, 4))

# for i, method in enumerate(linkage_methods):
#     linked = linkage(X, method=method)
#     dendrogram(linked, ax=axes[i])
#     axes[i].set_title("Agrupamiento jerarquico por " + method)
#     axes[i].set_xlabel("Indice de la muestra")
#     axes[i].set_ylabel("Distancia o similaridad")

for i, method in enumerate(linkage_methods):
    linked = linkage(X, method=method)
    dendrogram(linked, ax=axes[i])
    axes[i].set_title("Agrupamiento jerarquico por " + method)
    axes[i].set_xlabel("Indice de la muestra")
    axes[i].set_ylabel("Distancia o similaridad")

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
axes[-1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="rainbow")
axes[-1].set_title("Agrupamiento por KMeans")
axes[-1].set_xlabel("Indice de la muestra")
axes[-1].set_ylabel("Distancia o similaridad")

plt.tight_layout()
plt.show()
