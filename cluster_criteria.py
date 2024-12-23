# 测试聚类结果，以及可视化相关代码
import disjointSet # 手写并查集
import cluster # 手写层次聚类
import utils # 手写辅助函数
import matplotlib.pyplot as plt 
import numpy as np
# @utils.print_exec_time # 0.02s
def calcSSE(n, steps, p):
    '''给定聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的SSE列表 \n 复杂度 O(nα)≈O(n)'''
    dsu = disjointSet.DSU_SSE(n, p)
    sse = [0] * n
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        sse[-(i+2)] = dsu.sse
    return sse

def plotLine(seq, type_, metric, logScale=False):
    '''绘制折线图给定序列为seq，序列名字为type_，指标名字为metric，是否对y轴取对数logScale'''
    plt.plot(np.arange(1, len(seq) + 1), seq, marker='o')
    if logScale:
        plt.yscale('log')
    plt.title(f'{metric} {type_}')
    plt.xlabel('Number of Clusters')
    plt.ylabel(f'{metric}')
    plt.grid()

def calcSilhouette(n, steps, p):
    '''给定聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的轮廓系数列表 \n
    计算各点的轮廓系数Silhouette Coefficient的平均
    '''
    from sklearn.metrics import silhouette_score
    silhouette = [0] * n
    dsu = disjointSet.DSU_hard(n)
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        if abs(n-i)<=26:
            silhouette[-(i+2)] = silhouette_score(p, disjointSet.getClasses(dsu))
    return silhouette

def plotLines(metric):
    '''绘图展示四种聚类的指标(metric)可取SSE和silhouette'''
    p = utils.readCSV()
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i, type_ in enumerate(cluster.ALL_TYPES):
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        if metric == 'SSE':
            seq = calcSSE(p.shape[0], steps, p)
        elif metric == 'silhouette':
            seq = calcSilhouette(p.shape[0], steps, p)
        plt.subplot(2,2,i+1)
        plotLine(seq[:25], type_, 'SSE')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{metric}_partial.png')
# plotLines('SSE') 
plotLines('silhouette')

'''
def checkSilhouette():
    import numpy as np
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X) # [1 1 1 0 0 0]
    print(y_kmeans)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(f'{silhouette_avg}') # 0.7133477791749615
    s = 0
    for i in range(6):
        a = b = 0
        for j in range(6):
            dis = np.linalg.norm(X[i]-X[j])
            if y_kmeans[i] != y_kmeans[j]:
                b += dis
            else:
                a += dis
        a, b = a/2, b/3
        # s += 1 - a/b
        # s += (b-a)/max(b,a)
    print(s/6)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    print(y_kmeans)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(f'{silhouette_avg}') # 0.7133477791749615
    # from collections import defaultdict
    # for i in range(6):
    #     a = an = 0
    #     bl = defaultdict(list)
    #     for j in range(6):
    #         dis = np.linalg.norm(X[i]-X[j])
    #         if y_kmeans[i] == y_kmeans[j]:
    #             a += dis
    #             an += 1
    #         else:
    #             bl[y_kmeans[j]].append(dis)
    #     b = min( sum(bl[k])/len(bl[k]) for k in bl.keys())
    #     s += 1 - (a/an)/b
    # print(s/6)
    from sklearn.metrics import pairwise_distances
    def silhouette_coefficients(X, labels):
        n_samples = X.shape[0]
        distances = pairwise_distances(X)  # 计算距离矩阵
        silhouette_values = np.zeros(n_samples)

        for i in range(n_samples):
            # a(i): 数据点 i 到同簇内其他点的平均距离
            same_cluster = distances[i][labels == labels[i]]
            if len(same_cluster) > 1:  # 确保有多个点
                a_i = np.mean(same_cluster[same_cluster != 0])  # 排除自身
            else:
                a_i = 0  # 只有一个点时设为 0

            # b(i): 数据点 i 到最近簇的平均距离
            other_clusters = labels != labels[i]
            if np.any(other_clusters):  # 确保有其他簇
                b_i = np.min([np.mean(distances[i][other_clusters & (labels == c)]) for c in np.unique(labels[other_clusters])])
            else:
                b_i = 0  # 没有其他簇时设为 0

            # 计算 Silhouette Coefficient
            if a_i == b_i:
                silhouette_values[i] = 0  # 如果 a(i) 和 b(i) 相等，设为 0
            else:
                silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

        return silhouette_values
    print(silhouette_coefficients(X, y_kmeans))
    print(np.sum(silhouette_coefficients(X, y_kmeans))/6)
            
checkSilhouette()

def checkSilhouette2():
    import numpy as np
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
    from sklearn.cluster import KMeans
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X) # [1 1 1 0 0 0]
    print(y_kmeans)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(silhouette_avg)
    silhouette_vals = silhouette_samples(X, y_kmeans)
    print(silhouette_vals)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_kmeans = kmeans.fit_predict(X) # [1 1 1 0 0 0]
    print(y_kmeans)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(silhouette_avg)
    silhouette_vals = silhouette_samples(X, y_kmeans)
    print(silhouette_vals)
checkSilhouette2()
'''