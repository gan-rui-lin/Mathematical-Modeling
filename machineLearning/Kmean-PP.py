import numpy as np
from sklearn.cluster import k_means, KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def generate_sample_data(n_samples=4000, n_components=4, random_state=0):
    """
    生成样本数据用于聚类分析

    参数:
    n_samples: 样本数量
    n_components: 聚类中心数量
    random_state: 随机种子

    返回:
    X: 标准化后的特征数据
    """
    # 生成样本数据
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=n_components,
        cluster_std=0.60,
        random_state=random_state,
    )
    X = X[:, ::-1]

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, scaler


def silhouette_analysis(X, range_n_clusters=[2, 3, 4, 5, 6]):
    """
    使用轮廓分析选择 KMeans 聚类中的最佳簇数量

    参数:
    X: 特征数据
    range_n_clusters: 要测试的聚类数量范围

    返回:
    best_n_clusters: 最佳聚类数量
    silhouette_scores: 各聚类数量对应的轮廓分数
    """

    # 确保数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    silhouette_scores = {}

    for n_clusters in range_n_clusters:
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # 第一个子图是轮廓图
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # 初始化聚类器
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # 计算平均轮廓分数
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores[n_clusters] = silhouette_avg

        print(
            f"For n_clusters = {n_clusters}, "
            f"The average silhouette_score is : {silhouette_avg:.4f}"
        )

        # 计算每个样本的轮廓分数
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # 聚合属于簇 i 的样本的轮廓分数，并排序
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # 在轮廓图中间标注簇编号
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # 计算下一个图的 y_lower
            y_lower = y_upper + 10

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # 平均轮廓分数的垂直线
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 第二个子图显示实际形成的聚类
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # 标记聚类中心
        centers = clusterer.cluster_centers_
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()

    # 找到最佳聚类数量
    best_n_clusters = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\n最佳聚类数量: {best_n_clusters}")
    print(f"对应的轮廓分数: {silhouette_scores[best_n_clusters]:.4f}")

    return best_n_clusters, silhouette_scores


def perform_kmeans_clustering(X, n_clusters, random_state=0):
    """
    执行 K-means 聚类

    参数:
    X: 特征数据
    n_clusters: 聚类数量
    random_state: 随机种子

    返回:
    centroid: 聚类中心
    labels: 聚类标签
    inertia: 簇内平方和
    """
    # 执行 K-means 聚类
    centroid, labels, inertia = k_means(
        X,
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=random_state,
    )

    return centroid, labels, inertia


def visualize_clustering_results(X, labels, centroid, n_clusters):
    """
    可视化聚类结果

    参数:
    X: 特征数据
    labels: 聚类标签
    centroid: 聚类中心
    n_clusters: 聚类数量
    """
    plt.figure(figsize=(10, 8))

    # 绘制聚类结果
    colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
    plt.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # 标注聚类中心
    plt.scatter(
        centroid[:, 0],
        centroid[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    # 在中心点上标注簇编号
    for i, c in enumerate(centroid):
        plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    plt.title(f"K-Means Clustering Results (n_clusters={n_clusters})", fontsize=14)
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_clustering_summary(centroid, labels, inertia):
    """
    打印聚类结果摘要

    参数:
    centroid: 聚类中心
    labels: 聚类标签
    inertia: 簇内平方和
    """
    print("=" * 50)
    print("聚类结果摘要")
    print("=" * 50)
    print(f"聚类中心:\n{centroid}")
    print(f"\n聚类标签分布:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  簇 {cluster_id}: {count} 个样本")
    print(f"\n簇内平方和 (Inertia): {inertia:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    print("K-Means++ 聚类分析演示")
    print("=" * 60)

    # 1. 生成样本数据
    print("1. 生成样本数据...")
    X, scaler = generate_sample_data(n_samples=4000, n_components=4, random_state=0)
    print(f"   生成了 {X.shape[0]} 个样本，每个样本有 {X.shape[1]} 个特征")
    print(f"   数据已标准化，范围: [{X.min():.2f}, {X.max():.2f}]")

    # # 确保数据标准化
    # scaler = StandardScaler()
    # Y = scaler.fit_transform(X)
    # print(X == Y)

    # 2. 轮廓分析确定最佳聚类数量
    print("\n2. 进行轮廓分析确定最佳聚类数量...")
    range_n_clusters = [2, 3, 4, 5, 6]
    best_n_clusters, silhouette_scores = silhouette_analysis(X, range_n_clusters)

    # 3. 使用最佳聚类数量进行 K-means 聚类
    print(f"\n3. 使用最佳聚类数量 {best_n_clusters} 进行 K-means 聚类...")
    centroid, labels, inertia = perform_kmeans_clustering(X, best_n_clusters)

    # 4. 打印聚类结果摘要
    print_clustering_summary(centroid, labels, inertia)

    # 5. 可视化聚类结果
    print("\n4. 可视化聚类结果...")
    visualize_clustering_results(X, labels, centroid, best_n_clusters)

    # 6. 分析不同聚类数量的性能
    print("\n5. 不同聚类数量的轮廓分数对比:")
    print("-" * 40)
    for n_clusters, score in silhouette_scores.items():
        marker = " ← 最佳" if n_clusters == best_n_clusters else ""
        print(f"   {n_clusters} 个聚类: {score:.4f}{marker}")

    print(f"\n分析完成！最佳聚类数量为 {best_n_clusters} 个。")
