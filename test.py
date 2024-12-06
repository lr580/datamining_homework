import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 给定的距离矩阵
distance_matrix = np.array([
    [0, 0.24, 0.22, 0.37, 0.34, 0.23],
    [0.24, 0, 0.15, 0.20, 0.14, 0.25],
    [0.22, 0.15, 0, 0.15, 0.28, 0.11],
    [0.37, 0.20, 0.15, 0, 0.29, 0.22],
    [0.34, 0.14, 0.28, 0.29, 0, 0.39],
    [0.23, 0.25, 0.11, 0.22, 0.39, 0]
])

# 使用linkage函数进行层次聚类
# 计算成对距离并进行聚类
# 这里的距离矩阵需要转换为压缩格式
from scipy.spatial.distance import squareform

# 将完整的距离矩阵转换为压缩格式
condensed_distance_matrix = squareform(distance_matrix)

# 最小聚类（单链接）
linked_min = linkage(condensed_distance_matrix, method='single')
print(linked_min)
# 最大聚类（全链接）
linked_max = linkage(condensed_distance_matrix, method='complete')

# 组平均聚类
linked_avg = linkage(condensed_distance_matrix, method='average')

# 绘制树状图
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 最小聚类的树状图
dendrogram(linked_min, ax=axs[0])
axs[0].set_title('Dendrogram - Min Linkage (Single)')
axs[0].set_xlabel('Sample Index')
axs[0].set_ylabel('Distance')

# 最大聚类的树状图
dendrogram(linked_max, ax=axs[1])
axs[1].set_title('Dendrogram - Max Linkage (Complete)')
axs[1].set_xlabel('Sample Index')
axs[1].set_ylabel('Distance')

# 组平均聚类的树状图
dendrogram(linked_avg, ax=axs[2])
axs[2].set_title('Dendrogram - Average Linkage')
axs[2].set_xlabel('Sample Index')
axs[2].set_ylabel('Distance')

plt.tight_layout()
plt.show()