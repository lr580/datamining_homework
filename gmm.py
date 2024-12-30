# 手写实现的 GMM(高斯混合模型 Gaussian Mixture Models)
import numpy as np # 加速运算
import utils # 手写辅助函数，如读取数据集
class GMM:
    def __init__(self, k, init_strategy='random', seed=50, err=1e-4, max_iter=100):
        '''输入k:聚成k个类(k个二维正态(高斯)分布), 精度误差范围err，随机数种子seed，最大迭代次数max_iter\n
        初始化策略 strategy (参见init()方法)\n
        初始化一个高斯混合模型'''
        self.k = k
        self.err = err
        self.max_iter = max_iter
        self.seed = seed
        self.strategy = init_strategy
        
    def init_random(self, X):
        '''输入数据X[n][2], 随机初始化k个高斯分布的参数(μ,Σ,w)'''
        self.rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape # 点数，维度数(对本题一定是2)
        self.weights = np.ones(self.k) / self.k 
        self.means = X[self.rng.choice(n_samples, self.k, False)] 
        self.covariances = np.array([np.eye(n_features)] * self.k) 

    def calcParams(self, X, y, p):
        '''输入数据X[n][2], 标签y[n]，预测均值点p[n][2]，计算每个高斯分布的参数(μ,Σ,w)，内部辅助函数'''
        self.weights = np.ones(self.k) / self.k # z 权重，代表每个高斯分布占总数据的比例
        self.means = p # 每个高斯分布的均值
        self.covariances = np.array([np.cov(X[y == i].T) + 1e-6 * np.eye(X.shape[1]) for i in range(self.k)]) # 每个高斯分布的协方差矩阵

    def init_with_Kmeans(self, X, strategy='random'):
        '''输入数据X[n][2], 用Kmeans算法初始化k个高斯分布的参数(μ,Σ,w) \n
        若 strategy='random'，则随机选择Kmeans的初始中心点，否则('kmeans++')用kmeans++改进初始化'''
        kmeans = KMeans(self.k, self.seed, strategy)
        kmeans.fit(X)
        y = kmeans.predict(X)
        self.calcParams(X, y, kmeans.centroids)
        
    def init_with_kmedoids(self, X):
        '''输入数据X[n][2],用Kmedoids算法初始化k个高斯分布的参数(μ,Σ,w)\n
        未在正式代码使用，已废置'''
        kmedoids = KMedoids(n_clusters=self.k, random_state=self.seed)
        kmedoids.fit(X)
        self.means = kmedoids.medoids
        self.weights = np.ones(self.k) / self.k
        self.covariances = np.array([np.eye(X.shape[1])] * self.k)
        # 修改为下面的代码会让 KMeans/KMeans++受到影响，理由未知，所以放弃了修改
        # y = kmedoids.predict(X)
        # self.calcParams(X, y, kmedoids.medoids)

    ALL_INIT_STRAGEGY = ['random', 'kmedoids', 'kmeans', 'kmeans++']
    '''所有初始化策略'''

    def init(self, X, strategy='random'):
        '''输入数据X[n][2], 选择不同算法初始化k个高斯分布的参数(μ,Σ,w) \n
        - random: 随机初始化
        - kmedoids: 用kmedoids算法初始化
        - kmeans: 用Kmeans算法初始化
        - kmeans++: 用kmeans++改进初始化'''
        if strategy == 'random':
            self.init_random(X)
        elif strategy == 'kmeans':
            self.init_with_Kmeans(X)
        elif strategy == 'kmeans++':
            self.init_with_Kmeans(X, strategy='kmeans++')
            # self.init_with_Kmeans0(X)
        elif strategy == 'kmedoids':
            self.init_with_kmedoids(X)
        else:
            raise ValueError('Invalid strategy:', strategy)

    def fit(self, X):
        '''输入数据X[n][2], 对数据进行拟合学习，求出k个高斯分布的参数(μ,Σ,w)'''
        # self.init_random(X)
        # self.init_with_Kmeans(X)
        # self.init_with_kmedoids(X)
        self.init(X, self.strategy)
        log_likelihoods = [] # 每次迭代的最大似然估计
        for _ in range(self.max_iter): # EM 算法
            # print(_)
            responsibilities = self.e_step(X)  # E步
            self.m_step(X, responsibilities) # M步
            log_likelihood = self.log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.err:
                break # 变化小，可以跳出计算
        self.log_likelihoods = log_likelihoods # 用于调试输出

    def e_step(self, X):
        '''求出各个样本的后验概率：各成分加权密度函数的占该成分所有加权密度函数的占比。返回[k][n]向量'''
        weighted_probabilities = np.array([
            self.weights[i] * self.multivariate_gaussian(X, self.means[i], self.covariances[i]) for i in range(self.k)
        ])
        responsibilities = weighted_probabilities / weighted_probabilities.sum(axis=0)
        return responsibilities

    def m_step(self, X, responsibilities):
        '''利用最大似然估计法，用加权和方法求各参数(μ,Σ,w)'''
        N_i = responsibilities.sum(axis=1)
        self.weights = N_i / N_i.sum()
        self.means = np.dot(responsibilities, X) / N_i[:, np.newaxis]
        for i in range(self.k):
            diff = X - self.means[i]
            self.covariances[i] = np.dot(responsibilities[i] * diff.T, diff) / N_i[i]

    def multivariate_gaussian(self, X, mean, cov):
        '''给定二维高斯分布的参数和样本点X[n][2]，求这些样本点的概率密度p(x;)'''
        d = X.shape[1]
        numerator = np.exp(-0.5 * np.sum((X - mean) @ np.linalg.inv(cov) * (X - mean), axis=1))
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return numerator / denominator

    def log_likelihood(self, X):
        '''即把X计算所有k个高斯分布成分的加权和(似然函数)求对数，对所有样本求和'''
        return np.sum(np.log(np.sum([
            self.weights[i] * self.multivariate_gaussian(X, self.means[i], self.covariances[i]) for i in range(self.k)
        ], axis=0)))
    
    def predict(self, X):
        '''对输入X[n][2]，将其进行GMM聚类\n返回labels[n]∈[0,self.k)表示各店所属聚类类别'''
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=0)
    
    def print_params(self):
        '''调试输出每个高斯核的权重、均值、协方差，各迭代轮次的对数似然函数，以用于检验正确性'''
        print("Weights:", self.weights)
        print("Means:", self.means)
        print("Covariances:", self.covariances)
        if not hasattr(self, 'log_likelihoods'):
            return
        for i, x in enumerate(self.log_likelihoods):
            print(f'iter {i}: {x}')

    # @utils.print_exec_time
    def init_with_Kmeans0(self, X, seed=50):
        '''输入数据X[n][2], 用Kmeans算法初始化k个高斯分布的参数(μ,Σ,w) \n
        未在正式代码使用，已废置'''
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.k, random_state=seed)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_
        self.weights = np.ones(self.k) / self.k
        # self.covariances = np.array([np.eye(X.shape[1])] * self.k)
        self.covariances = np.array([np.cov(X[kmeans.labels_ == i].T) + 1e-6 * np.eye(X.shape[1]) for i in range(self.k)])


class KMeans:
    '''手写Kmeans聚类'''
    def __init__(self, n_clusters, seed=42, init_strategy='random', max_iter=300, err=1e-4):
        '''初始化参数，聚类数目n_clusters, 随机种子seed，最大迭代次数max_iter，err收敛误差率\n
        init_strategy初始化策略，若为'random'随机初始，否则用kmeans++初始化'''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.err = err
        self.centroids = None
        self.random_state = seed
        self.init_strategy = init_strategy

    def init_random(self, X):
        '''随机初始化质心'''
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def init_improved(self, X):
        '''Kmeans++初始化质心'''
        n_samples, _ = X.shape
        centroids = np.empty((self.n_clusters, X.shape[1]))
        self.centroids = centroids
        centroids[0] = X[np.random.choice(n_samples)]
        for k in range(1, self.n_clusters):
            distances = self.compute_distances(X)
            min_distances = np.min(distances, axis=1)
            probabilities = min_distances / np.sum(min_distances)
            centroids[k] = X[np.random.choice(n_samples, p=probabilities)]
        return centroids

    def init(self, X):
        '''初始化质心'''
        np.random.seed(self.random_state)
        if self.init_strategy == 'random':
            self.init_random(X)
        else: # 'kmeans++'
            self.init_improved(X)

    def fit(self, X):
        '''迭代拟合求质心'''
        self.init(X)
        for _ in range(self.max_iter):
            distances = self.compute_distances(X) # 每个点到质心的距离
            labels = np.argmin(distances, axis=1) # 新标签
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)]) # 新质心
            if np.all(np.abs(new_centroids - self.centroids) < self.err):
                break # 检查收敛
            self.centroids = new_centroids

    def compute_distances(self, X):
        '''求每个点到质心的距离'''
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def predict(self, X):
        '''为每个点求出分类标签'''
        distances = self.compute_distances(X)
        return np.argmin(distances, axis=1)

class KMedoids:
    '''手写 Kmedoids 聚类算法'''
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        '''初始化，n_clusters聚类数，max_iter迭代数，random_state随机种子'''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        '''拟合'''
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        # 随机选择初始的 medoids
        initial_medoids_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[initial_medoids_indices]
        for _ in range(self.max_iter):
            labels = self.assign_labels(X)
            new_medoids = self.update_medoids(X, labels)
            if np.array_equal(new_medoids, self.medoids):
                break # 收敛
            self.medoids = new_medoids
        return self

    def assign_labels(self, X):
        '''计算每个样本到所有 medoids 的距离，并分配标签'''
        distances = np.array([[np.linalg.norm(x - m) for m in self.medoids] for x in X])
        return np.argmin(distances, axis=1)

    def update_medoids(self, X, labels):
        '''更新medoids'''
        new_medoids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # 选择使得到该簇中所有点的距离之和最小的点
                distances = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
                new_medoids[i] = cluster_points[np.argmin(distances)]
        return new_medoids

    def predict(self, X):
        '''进行预测，返回分类标签X'''
        return self.assign_labels(X)

# @utils.print_exec_time
def GMMcluster(X=None, k=15, strategy='kmeans++', seed=50):
    '''使用手写GMM聚类，对数据集X(8gau.txt)进行聚类，聚成k类 \n
    初始化策略为 strategy, 参见 GMM.init() 函数描述 \n
    返回聚类结果y和模型本身'''
    if X is None:
        X = utils.readCSV()
    gmm = GMM(k, strategy, seed)
    X0 = utils.z_score(X) # 手写标准化
    gmm.fit(X0)
    y = gmm.predict(X0)
    return y, gmm

# 下面是测试代码，主要用于检验代码正确性
@utils.print_exec_time
def test_kmeans():
    '''检验聚类正确性，未在正式代码使用'''
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=5000, centers=15, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=15)
    kmeans.fit(X)
    print(kmeans.centroids)
# test_kmeans()

def test_multivariate_gaussian():
    '''测试高斯分布函数的矩阵计算，未在正式代码使用'''
    np.random.seed(0)  # 设置随机种子以便可重复
    mean = np.array([0, 0])  # 均值
    cov = np.array([[1, 0.5], [0.5, 1]])  # 协方差矩阵
    X = np.random.multivariate_normal(mean, cov, size=5)  # 生成5个样本点
    gmm = GMM(1)
    res = gmm.multivariate_gaussian(X, mean, cov)
    print(res)
# test_multivariate_gaussian()
    
# print(GMM.ALL_INIT_STRAGEGY)

def check_KMeans(seed=996):
    '''检验KMeans聚类正确性，未在正式代码使用 \n
    经过大量随机数实验，seed=996时，手写和标准KMeans的聚类结果一致 \n
    且其计算出来的均值和协方差也是几乎一样的，未在正式代码使用'''
    X = utils.readCSV()
    X0 = utils.z_score(X)
    # X0 = X
    gmm = GMM(15, 'kmeans++', seed)
    gmm.init_with_Kmeans(X0, 'kmeans++')
    mean1 = gmm.means
    cov1 = gmm.covariances
    idx1 = mean1[:, 0].argsort()
    mean1 = mean1[idx1]
    cov1 = cov1[idx1]
    
    gmm.init_with_Kmeans0(X0)
    mean2 = gmm.means
    cov2 = gmm.covariances
    idx2 = mean2[:, 0].argsort()
    mean2 = mean2[idx2]
    cov2 = cov2[idx2]
   
    for i in range(mean1.shape[0]):
        print(mean1[i], mean2[i])
    for i in range(mean1.shape[0]):
        print(cov1[i].flatten(), '\n', cov2[i].flatten(), '\n')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(mean1[:, 0], mean1[:, 1], color='blue', label='KMeans++ Means', marker='o')
    plt.scatter(mean2[:, 0], mean2[:, 1], color='red', label='KMeans Means', marker='x')
    
    plt.title('Comparison of KMeans Means')
    plt.xlabel('Mean 1')
    plt.ylabel('Mean 2')
    plt.legend()
    plt.grid()
    plt.show()
# check_KMeans(8146)