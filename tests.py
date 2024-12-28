import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(0)
n_samples = 300
X = np.vstack([np.random.normal(loc, 0.5, (n_samples, 2)) for loc in [(-2, -2), (2, 2), (0, 3)]])

gmm = GaussianMixture(n_components=3, random_state=0)
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