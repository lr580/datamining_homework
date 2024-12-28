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
    gmm = GMM(3)
    gmm.fit(X)
    y = gmm.predict(X)
    gmm.print_params()
    cluster_criteria.plotClusterResults(X, y, 'GMM')
    plt.show()
gmm_generate2()