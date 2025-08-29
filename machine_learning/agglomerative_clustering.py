from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


n_samples = 4000
n_components = 4

X, _ = make_blobs(
    n_samples=n_samples,
    centers=n_components,
    cluster_std=0.60,
    random_state=0,
)
X = X[:, ::-1]

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

Z = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.show()

clustering = AgglomerativeClustering(
    n_clusters=4, metric="euclidean", linkage="ward", compute_distances=True
).fit(X)
# print(clustering.labels_)
label = clustering.labels_
n_clusters = clustering.n_clusters_
print(n_clusters)

# 绘制散点图
plt.figure(figsize=(8, 6))
colors = cm.get_cmap("nipy_spectral")(label.astype(float) / n_clusters)
print(colors)
plt.scatter(
    X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
)

plt.title("The visualization of the clustered data.")
plt.xlabel("Feature space for the 1st feature")
plt.ylabel("Feature space for the 2nd feature")

plt.show()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(clustering, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
