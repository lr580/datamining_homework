# 测试聚类结果，以及可视化相关代码
import disjointSet # 手写并查集
import cluster # 手写层次聚类
import utils # 手写辅助函数
import matplotlib.pyplot as plt 
import numpy as np
# @utils.print_exec_time # 0.02s
def calcSSEs(n, steps, p):
    '''给定聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的SSE列表 \n 复杂度 O(nα)≈O(n)'''
    dsu = disjointSet.DSU_SSE(n, p)
    sse = [0] * n
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        sse[-(i+2)] = dsu.sse
    return sse

# @utils.print_exec_time
def calcSilhouette(dis, labels):
    '''给定距离矩阵dis和每个店所属的labels数组，求轮廓系数，复杂度 O(n^2) \n 
    表现性能：n=5000,2.37s 与标准解法处于同一复杂度'''
    n = len(labels)
    unique_labels = np.unique(labels)
    count_labels = np.bincount(labels)
    silhouette = np.zeros(n)
    for i in range(n):
        label_i = labels[i]
        if count_labels[label_i] == 1:
            silhouette[i] = 0
            continue
        a_i = np.sum(dis[i, labels == label_i]) / (count_labels[label_i] - 1)  # 除掉自己
        b_i = np.inf
        for label_j in unique_labels:
            if label_j != label_i:
                b_i = min(b_i, np.mean(dis[i, labels == label_j]))
        
        silhouette[i] = (b_i - a_i) / max(a_i, b_i)
    # print(silhouette)
    return np.mean(silhouette)

def calcSilhouettes(n, steps, p, calcLim=25):
    '''给定聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的轮廓系数列表，calcLim 是最多多少类时计算轮廓系数 \n
    计算各点的轮廓系数Silhouette Coefficient的平均
    '''
    silhouette = [0] * n
    dsu = disjointSet.DSU_hard(n)
    distance_matrix = utils.getDisMatrix(p)
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        if 1<dsu.num<=calcLim:
            # 做 4 个图，各 25 次计算，理论最快需要 40s
            labels = disjointSet.getClasses(dsu)
            silhouette[-(i+2)] = calcSilhouette(distance_matrix, labels)
    return silhouette

def plotLine(seq, type_, metric, logScale=False):
    '''绘制折线图给定序列为seq，序列名字为type_，指标名字为metric，是否对y轴取对数logScale'''
    plt.plot(np.arange(2, len(seq) + 2), seq, marker='o')
    if logScale:
        plt.yscale('log')
    plt.title(f'{metric} {type_}')
    plt.xlabel('Number of Clusters')
    plt.ylabel(f'{metric}')
    plt.grid()
    
def plotDoubleLines(y1, y2, x, y1name, y2name, ax1, type_):
    '''绘制双Y轴折线图给定两个序列为y1,y2，x轴为x，两个序列名字为y1name,y2name，图标题为type_'''
    ax1.set_title(f'{type_} Cluster') 
    ax1.set_xlabel('Number of Clusters')
    c1, c2 = 'lightcoral', 'steelblue' # 绘图颜色(teal, orange)
    
    ax1.plot(x, y1, marker='o', label=y1name, color=c1)
    ax1.set_ylabel(y1name, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    
    ax2 = ax1.twinx()
    ax2.plot(x, y2, marker='o', label=y2name, color=c2)
    ax2.set_ylabel(y2name, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    # ax1.legend(loc='upper left')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.86))
    ax1.grid()

# @utils.print_exec_time
def plotLines(metric):
    '''绘图展示四种聚类的指标(metric)可取SSE和silhouette和both'''
    p = utils.readCSV()
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    if metric == 'both':
        axs = axs.flatten()  # 将子图数组扁平化，方便索引
    plt.suptitle('SSE and Silhouette Coefficient of Clustering')
    for i, type_ in enumerate(cluster.ALL_TYPES):
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        plt.subplot(2,2,i+1)
        if metric == 'SSE':
            seq = calcSSEs(p.shape[0], steps, p)
            plotLine(seq[1:25], type_, metric)
        if metric == 'silhouette':
            seq = calcSilhouettes(p.shape[0], steps, p)
            plotLine(seq[1:25], type_, metric)
        if metric == 'both':
            seq1= calcSSEs(p.shape[0], steps, p)
            seq2 = calcSilhouettes(p.shape[0], steps, p)
            plotDoubleLines(seq1[1:25], seq2[1:25], np.arange(2, 26), 'SSE', 'silhouette', axs[i], type_)
        
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{metric}_partial.png')
# plotLines('SSE') 
# plotLines('silhouette')
plotLines('both')

def plotClusterResults(p, labels, dest_path, cmap='tab20'):
    '''给定点集p[n][2], 聚类结果labels[n]，绘制聚类结果图，保存于路径dest_path；颜色colormap为cmap，如'tab20' '''
    plt.scatter(p[:, 0], p[:, 1], c=labels, cmap=cmap)
    plt.title('Clustering Results')
    # plt.colorbar(label='class')
    plt.savefig(dest_path)
    
def plotAllTypesCluster(k=15):
    '''绘制四种层次聚类结果,聚成k类'''
    p = utils.readCSV()
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    
    for i, type_ in enumerate(cluster.ALL_TYPES):
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        plt.subplot(2,2,i+1)
        

# 下面代码没有在正式部分使用
# @utils.print_exec_time
def checkSilhouette(p, labels, score):
    '''检验计算正确性，设算出来系数是score，检验其是否正确\n未在正式代码使用，只用于测试'''
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
    answer = silhouette_score(p, labels)
    # print(silhouette_samples(p, labels))
    assert abs(answer - score) < 1e-6, f'answer: {answer}, score: {score}'
    
def testSilhouette1():
    '''检查正确性，未在正式代码使用，只用于测试 \n 检测轮廓系数的计算'''
    p = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(p)
    # print(labels)
    dis = utils.getDisMatrix(p)
    score = calcSilhouette(dis, labels)
    checkSilhouette(p, labels, score)
# testSilhouette1()
    
def testSilhouette2():
    '''检查正确性，未在正式代码使用，只用于测试 \n 检测轮廓系数的计算'''
    p = np.random.rand(5000, 2)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=25, random_state=0)
    labels = kmeans.fit_predict(p)
    dis = utils.getDisMatrix(p)
    score = calcSilhouette(dis, labels)
    checkSilhouette(p, labels, score)
# testSilhouette2()