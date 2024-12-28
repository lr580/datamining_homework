# 手写实现的 GMM(高斯混合模型)
import numpy as np
class GMM:
    def __init__(self, k, err=1e-4, max_iter=100):
        '''输入k:聚成k个类(k个二维正态(高斯)分布), 精度误差范围err，最大迭代次数max_iter \n
        初始化一个高斯混合模型'''
        self.k = k
        self.err = err
        self.max_iter = max_iter

    def fit(self, X):
        '''输入数据X[n][2], 对数据进行拟合'''
        n_samples, n_features = X.shape # 点数，维度数(对本题一定是2)
        self.weights = np.ones(self.k) / self.k # z 权重，代表每个高斯分布占总数据的比例
        self.means = X[np.random.choice(n_samples, self.k, False)] # 每个高斯分布的均值
        self.covariances = np.array([np.eye(n_features)] * self.k) # 每个高斯分布的协方差矩阵
        log_likelihoods = [] # 每次迭代的最大似然估计
        for _ in range(self.max_iter): # EM 算法
            responsibilities = self.e_step(X)  # E步
            self.m_step(X, responsibilities) # M步
            log_likelihood = self._log_likelihood(X)
            log_likelihoods.append(log_likelihood)
            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.err:
                break # 变化小，可以跳出计算

    def e_step(self, X):
        weighted_probabilities = np.array([
            self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
            for i in range(self.k)
        ])
        responsibilities = weighted_probabilities / weighted_probabilities.sum(axis=0)
        return responsibilities

    def m_step(self, X, responsibilities):
        N_i = responsibilities.sum(axis=1)

        self.weights = N_i / N_i.sum()
        self.means = np.dot(responsibilities, X) / N_i[:, np.newaxis]

        for i in range(self.k):
            diff = X - self.means[i]
            self.covariances[i] = np.dot(responsibilities[i] * diff.T, diff) / N_i[i]

    def _multivariate_gaussian(self, X, mean, cov):
        d = X.shape[1]
        numerator = np.exp(-0.5 * np.sum((X - mean) @ np.linalg.inv(cov) * (X - mean), axis=1))
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return numerator / denominator

    def _log_likelihood(self, X):
        return np.sum(np.log(np.sum([
            self.weights[i] * self._multivariate_gaussian(X, self.means[i], self.covariances[i])
            for i in range(self.k)
        ], axis=0)))
    
    def predict(self, X):
        '''对输入X[n][2]，将其进行GMM聚类\n返回labels[n]∈[0,self.k)表示各店所属聚类类别'''
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=0)
    
    def print_params(self):
        '''调试输出每个高斯核的权重、均值、协方差，以用于检验正确性'''
        print("Weights:", self.weights)
        print("Means:", self.means)
        print("Covariances:", self.covariances)