from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import utils
import cluster_criteria
# from test_gmm import GMM
from gmm import GMM

def gmm_trail(X, k):

    # np.random.seed(0)
    # n_samples = 300
    # X = np.vstack([np.random.normal(loc, 0.5, (n_samples, 2)) for loc in [(-2, -2), (2, 2), (0, 3)]])

    gmm = GaussianMixture(n_components=k, random_state=0)
    gmm.fit(X)
    for i in range(gmm.n_components):
        print(f"高斯核 {i + 1}:")
        print(f"  权重: {gmm.weights_[i]}")
        print(f"  均值: {gmm.means_[i]}")
        print(f"  协方差矩阵: \n{gmm.covariances_[i]}\n")

    labels = gmm.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
    plt.title('GMM Clustering Results')
    plt.show()



def gmm_generate():
    # 设置随机种子
    np.random.seed(42)

    # 定义高斯混合模型的参数
    n_components = 3  # 高斯分布的个数
    n_samples = 500   # 生成的数据点数量

    # 定义每个高斯成分的均值和协方差
    means = np.array([[0, 0], [5, 5], [5, 0]])
    covariances = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]])

    # 创建空的列表来存储数据和标签
    data = []
    labels = []

    # 生成每个成分的数据
    for i in range(n_components):
        component_data = np.random.multivariate_normal(means[i], covariances[i], size=n_samples // n_components)
        data.append(component_data)
        labels.append(np.full((n_samples // n_components,), i))  # 添加标签

    # 合并数据和标签
    data = np.vstack(data)
    labels = np.concatenate(labels)

    # 打乱数据顺序
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # 绘制结果
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, cmap='viridis')
    plt.title("2D Gaussian Mixture Data Points with Labels")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.colorbar(label='Component Label')
    plt.show()

    return data, labels  # 返回数据和标签

# 调用函数生成数据
# data, labels = gmm_generate()

def gmm_generate2():
    X, y_ = utils.getGMMsamplePoints(*utils.getGMMsampleParams())
    # gmm = GaussianMixture(n_components=3, random_state=0)
    # gmm.fit(X)
    # y = gmm.predict(X)
    # for i in range(gmm.n_components):
    #     print(i + 1)
    #     print(gmm.weights_[i])
    #     print(gmm.means_[i])
    #     print(gmm.covariances_[i])
    X0 = utils.z_score(X)
    gmm = GMM(3)
    gmm.fit(X0)
    y = gmm.predict(X0)
    gmm.print_params()
    cluster_criteria.plotClusterResults(X, y, 'GMM')
    plt.show()
# gmm_generate2()

def norm1dCase():
    from scipy.stats import norm

    # 定义正态分布的参数
    mu = 0      # 均值
    sigma = 1   # 标准差

    # 定义样本值
    x0 = 0.5

    # 生成x值用于绘图
    x = np.linspace(-4, 4, 100)
    pdf = norm.pdf(x, mu, sigma)

    # 计算特定样本值的概率密度
    pdf_x0 = norm.pdf(x0, mu, sigma)

    # 打印结果
    print(f"样本值 {x0} 的概率密度为: {pdf_x0}")

    # 绘制正态分布图
    plt.plot(x, pdf, label='PDF', color='blue')
    plt.axvline(x0, color='red', linestyle='--', label=f'x= {x0}')
    plt.scatter([x0], [pdf_x0], color='red')  # 样本值的概率密度点
    plt.title('PDF of Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid()
    plt.show()
# norm1dCase()

def norm2dCase():
    from scipy.stats import multivariate_normal

    # 定义二维正态分布的参数
    mu = [0, 0]  # 均值向量
    sigma = [[1, 0.5], [0.5, 1]]  # 协方差矩阵

    # 生成网格数据
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # 计算二维概率密度函数
    pdf = multivariate_normal.pdf(pos, mean=mu, cov=sigma)

    # 选择样本点
    sample_point = np.array([1, 1])
    pdf_sample = multivariate_normal.pdf(sample_point, mean=mu, cov=sigma)

    # 打印结果
    print(f"样本点 {sample_point} 的概率密度为: {pdf_sample}")

    # 绘制二维正态分布图
    plt.contourf(X, Y, pdf, levels=50, cmap='viridis')
    plt.colorbar(label='概率密度')
    plt.scatter(*sample_point, color='red')  # 样本点
    plt.title('二维正态分布')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.grid()
    plt.show()
# norm2dCase()

def random_test():
    # 使用固定种子生成随机数
    fixed_seed = 42
    rng_fixed = np.random.default_rng(fixed_seed)
    print(rng_fixed.random(5)) 
    print(np.random.random(5))
# random_test()

# def test_kmed():
#     from sklearn_extra.cluster import KMedoids
#     import numpy as np

#     # 生成示例数据
#     data = np.random.rand(100, 2)

#     # K-medoids 聚类
#     kmedoids = KMedoids(n_clusters=3, random_state=0).fit(data)

#     # 获取聚类结果
#     labels = kmedoids.labels_
#     medoids = kmedoids.cluster_centers_

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

def test_kmed():
    # 生成示例数据
    np.random.seed(42)
    X = np.random.rand(100, 2)

    # K-medoids 聚类
    kmedoids = KMedoids(n_clusters=3, random_state=42)
    kmedoids.fit(X)
    labels = kmedoids.predict(X)

    print("Medoids:", kmedoids.medoids)
    print("Labels:", labels)
# test_kmed()

'''
def init_with_KNN(self, X):
    输入数据X[n][2], 用KNN算法初始化k个高斯分布的参数(μ,Σ,w)\n
    未在正式代码使用，已废置
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=self.k)
    labels = np.arange(self.k)
    knn.fit(X, labels)
    self.means = knn.kneighbors(X, return_distance=False).mean(axis=1)
    self.weights = np.ones(self.k) / self.k 
    self.covariances = np.array([np.eye(X.shape[1])] * self.k)
'''