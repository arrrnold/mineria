# dendograma

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers=3, random_state=42)
linked = linkage(X, method="single")

dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.title("Dendograma de agrupamiento jerarquico")
plt.xlabel("Indice del punto de datos")
plt.ylabel("Distancia")
plt.show()


