# 手写 Kmediods 聚类算法，已废置
import numpy as np
class KMedoids:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # 随机选择初始的 medoids
        initial_medoids_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[initial_medoids_indices]

        for _ in range(self.max_iter):
            # 分配步骤
            labels = self._assign_labels(X)

            # 更新步骤
            new_medoids = self._update_medoids(X, labels)

            # 如果 medoids 没有变化，则停止迭代
            if np.array_equal(new_medoids, self.medoids):
                break
            self.medoids = new_medoids

        return self

    def _assign_labels(self, X):
        # 计算每个样本到所有 medoids 的距离，并分配标签
        distances = np.array([[np.linalg.norm(x - m) for m in self.medoids] for x in X])
        return np.argmin(distances, axis=1)

    def _update_medoids(self, X, labels):
        new_medoids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            # 获取当前簇的所有样本
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # 选择使得到该簇中所有点的距离之和最小的点
                distances = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
                new_medoids[i] = cluster_points[np.argmin(distances)]
        return new_medoids

    def predict(self, X):
        return self._assign_labels(X)