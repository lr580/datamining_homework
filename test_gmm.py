import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class GMM:
    def __init__(self, n_components, tol=1e-4, max_iter=100):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, False)]
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        log_likelihoods = []

        for _ in range(self.max_iter):
            # E步
            responsibilities = self._e_step(X)

            # M步
            self._m_step(X, responsibilities)

            # 计算对数似然
            log_likelihood = self._log_likelihood(X)
            log_likelihoods.append(log_likelihood)

            # 检查收敛
            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:
                break

    def _e_step(self, X):
        weighted_probabilities = np.array([
            self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        ])
        responsibilities = weighted_probabilities / weighted_probabilities.sum(axis=0)
        return responsibilities

    def _m_step(self, X, responsibilities):
        N_k = responsibilities.sum(axis=1)

        self.weights = N_k / N_k.sum()
        self.means = np.dot(responsibilities, X) / N_k[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[k] * diff.T, diff) / N_k[k]

    def _multivariate_gaussian(self, X, mean, cov):
        d = X.shape[1]
        numerator = np.exp(-0.5 * np.sum((X - mean) @ np.linalg.inv(cov) * (X - mean), axis=1))
        denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
        return numerator / denominator

    def _log_likelihood(self, X):
        return np.sum(np.log(np.sum([
            self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        ], axis=0)))
    
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=0)
    
    def print_params(self):
        print("Weights:", self.weights)
        print("Means:", self.means)
        print("Covariances:", self.covariances)

def test_():
    # 示例数据
    X, _ = make_moons(n_samples=300, noise=0.05)

    # 训练GMM模型
    gmm = GMM(n_components=2)
    gmm.fit(X)

    # 进行预测
    predictions = gmm.predict(X)
    gmm.print_params()

    # 可视化结果
    plt.scatter(X[:, 0], X[:, 1], c=predictions, s=40, cmap='viridis')
    plt.title("GMM Clustering with Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()