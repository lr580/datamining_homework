# 测试聚类结果，以及可视化相关代码
import disjointSet # 手写并查集
import cluster # 手写层次聚类
import gmm # 手写GMM
import utils # 手写辅助函数
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.patches import Patch
# @utils.print_exec_time # 0.02s
def calcSSEs(n, steps, p):
    '''给定层次聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的SSE列表 \n 复杂度 O(nα)≈O(n)'''
    dsu = disjointSet.DSU_SSE(n, p)
    sse = [0] * n
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        sse[-(i+2)] = dsu.sse
    return sse

def calcSSE(p, labels, k):
    '''给定点集p和每个店所属的labels数组和类别数k，求SSE，复杂度为 O(n)，用于 GMM'''
    sse = 0
    for i in range(k):
        cluster_points = p[labels == i]
        centroid = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - centroid) ** 2)
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
    计算各点的轮廓系数Silhouette Coefficient的平均'''
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
    
def plotDoubleLines(y1, y2, x, y1name, y2name, ax1, type_, bbox=(1,.8)):
    '''绘制双Y轴折线图给定两个序列为y1,y2，x轴为x，两个序列名字为y1name,y2name，图标题为type_，图例位置bbox'''
    type_ = type_ if type_.isupper() else type_.title()
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
    # ax1.legend(loc='upper right')
    # ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.86))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = [Patch(color=c1, alpha=1, label=y1name), Patch(color=c2, alpha=1, label=y2name)]
    ax2.legend(handles=handles, loc='upper right', bbox_to_anchor=bbox)
    ax1.grid()

# @utils.print_exec_time
def plotLines(metric, show=False):
    '''绘图展示四种聚类的指标(metric)可取SSE和silhouette和both；show区分是直接展示(True)还是保存到本地(False)'''
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
    if show:
        plt.show()
    else:
        plt.savefig(f'{metric}_partial.png')
# plotLines('SSE') 
# plotLines('silhouette')
# plotLines('both')

def plotClusterResults(p, labels, type_, cmap='tab20'):
    '''给定点集p[n][2], 聚类结果labels[n]，绘制聚类结果图颜色colormap为cmap，如'tab20' '''
    plt.scatter(p[:, 0], p[:, 1], c=labels, cmap=cmap)
    type_ = type_[0].upper() + type_[1:] # 格式化首字符大写
    plt.title(f'{type_} Cluster')
    # plt.colorbar(label='class')
    
def plotAllTypesCluster(dest_path='cluster_results.png', k=15, show=False):
    '''绘制四种层次聚类结果,聚成k类; 保存于路径dest_path；show区分是直接展示(True)还是保存到本地(False)'''
    p = utils.readCSV()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plt.suptitle('Clustering Results of Different Types')
    for i, type_ in enumerate(cluster.ALL_TYPES):
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        plt.subplot(2,2,i+1)
        labels = cluster.ClusterFromSteps(p.shape[0], steps, k)
        plotClusterResults(p, labels, type_, cmap='tab20')
    if show: 
        plt.show()
    else:
        plt.savefig(dest_path)
# plotAllTypesCluster('cluster_results.png')

def plotSteps(steps, type_):
    '''绘制层次聚类结果的层次聚类步骤，steps为原始聚类步骤，type_是聚类类型，需要一定的时间渲染图像'''
    from scipy.cluster.hierarchy import dendrogram
    steps = cluster.BuildPlottingSteps(steps)
    # if k > 1:
    #     steps = steps[:-(k+1),:]
    plt.figure(figsize=(10, 7))
    dendrogram(steps, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram of '+f'{type_} Cluster'.title())
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.xticks([])  # 隐藏 x 轴的横坐标
    plt.savefig(f'{type_}_clustering_steps.png')

def plotAllSteps():
    '''绘制所有四种层次聚类的聚类过程图；是否合并成一张图'''
    for type_ in cluster.ALL_TYPES:
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        plotSteps(steps, type_)
# plotAllSteps()

def plotLines_GMM(seed=8146, show=False):
    '''绘图展示GMM聚类为k=15类的结果，测试不同的初始化策略'''
    p = utils.readCSV()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.flatten()
    plt.suptitle('Different Init Strategies of GMM Clustering')
    for i, strategy in enumerate(gmm.GMM.ALL_INIT_STRAGEGY):
        label, model = gmm.GMMcluster(p, 15, strategy, seed)
        plt.subplot(2,2,i+1)
        plotClusterResults(p, label, f'GMM (With {strategy.title()} Init) ')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'GMM_different_strategy.png')
# for i in range(20): # 寻找较优的随机参数
#     seed = np.random.randint(1000, 10000)
#     print(seed)
#     plotLines_GMM(seed, True)
# plotLines_GMM(8146, True)
# plotLines_GMM(8146, False)

K_LIST = [2, 4, 6, 8, 10, 12, 15, 17, 20]
'''用于测试GMM的不同分类数目k'''

def plotLines_GMM_k(seed=8208, show=False):
    '''绘图展示GMM聚类为不同k的结果'''
    p = utils.readCSV()
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    plt.suptitle('Different K for GMM Clustering')
    for i, k in enumerate(K_LIST):
        label, model = gmm.GMMcluster(p, k, 'kmeans++', seed)
        plt.subplot(3,3,i+1)
        plotClusterResults(p, label, f'GMM (K={k})')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(f'GMM_different_k.png')
# seed = np.random.randint(1000, 10000)
# print(seed)
# plotLines_GMM_k(seed, True)
# plotLines_GMM_k(8208, True)
# plotLines_GMM_k(8208, False)

def plotSSE_Silhouette_GMM_k(seed=8208, show=False):
    '''绘图展示GMM聚类为不同k的结指标'''
    p = utils.readCSV()
    p_dist = utils.getDisMatrix(p)
    # plt.suptitle('SSE and Silhoutte Coefficient of Different K for GMM Clustering')
    sses, silhouettes = [], []
    for i, k in enumerate(K_LIST):
        label, model = gmm.GMMcluster(p, k, seed=seed)
        sses.append(calcSSE(p, label, k))
        silhouettes.append(calcSilhouette(p_dist, label))
    fig, ax1 = plt.subplots()
    plotDoubleLines(sses, silhouettes, K_LIST, 'SSE', 'Silhouette', ax1, 'GMM', (1,1))
    if show:
        plt.show()
    else:
        plt.savefig(f'GMM_different_k_sse_silhouette.png')
# plotSSE_Silhouette_GMM_k(8208, True)
# plotSSE_Silhouette_GMM_k(8208, False)

def plotWardVsGMM(seed=8914, show=False):
    '''绘图展示Ward和GMM的对比'''
    p = utils.readCSV()
    with open('steps_ward.txt', 'r') as f:
        steps = eval(f.read())
    label_ward = cluster.ClusterFromSteps(p.shape[0], steps, 15)
    # sse_ward = calcSSE(p, label_ward, 15)
    # silhouette_ward = calcSilhouette(p_dist, label_ward)
    label_gmm, model = gmm.GMMcluster(p, 15, 'kmeans++', seed)
    # sse_gmm = calcSSE(p, label_gmm, 15)
    # silhouette_gmm = calcSilhouette(p_dist, label_gmm)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    plt.suptitle('Hierarchical Cluster VS GMM Cluster')
    plt.subplot(1, 2, 1)
    plotClusterResults(p, label_ward, 'Ward Hierarchical')
    plt.subplot(1, 2, 2)
    plotClusterResults(p, label_gmm, 'GMM (KMeans++ Init)')
    if show:
        plt.show()
    else:
        plt.savefig(f'Ward_vs_GMM.png')
# seed = np.random.randint(1000, 10000)
# print(seed)
# plotWardVsGMM(8914, True)
# plotWardVsGMM(8914, False)

def compareWardAndGMM(seed=8914, show=False):
    '''对比Ward和GMM的聚类结果'''
    p = utils.readCSV()
    p_dist = utils.getDisMatrix(p)
    with open('steps_ward.txt', 'r') as f:
        steps = eval(f.read())
    label_ward = cluster.ClusterFromSteps(p.shape[0], steps, 15)
    sse_ward = calcSSE(p, label_ward, 15)
    silhouette_ward = calcSilhouette(p_dist, label_ward)
    label_gmm, model = gmm.GMMcluster(p, 15, 'kmeans++', seed)
    sse_gmm = calcSSE(p, label_gmm, 15)
    silhouette_gmm = calcSilhouette(p_dist, label_gmm)
    # print(sse_ward, sse_gmm)
    # print(silhouette_ward, silhouette_gmm)
    print("{:<20} {:<20}".format("SSE Ward", "SSE GMM"))
    print("{:<20} {:<20}".format(sse_ward, sse_gmm))
    print("{:<20} {:<20}".format("Silhouette Ward", "Silhouette GMM"))
    print("{:<20} {:<20}".format(silhouette_ward, silhouette_gmm))

    labels = ['SSE', 'Silhoutte Coefficient']
    fig, ax1 = plt.subplots()
    width = 0.35 # 柱子宽
    x = np.arange(2)
    ax1.bar(x - width/2, [sse_ward, sse_gmm], width, label='Ward Hierarchical Cluster', color='b')
    ax1.set_ylabel('SSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()  # 共享x轴
    ax2.bar(x + width/2, [silhouette_ward, silhouette_gmm], width, label='GMM (KMeans++ Init) Cluster', color='r')
    ax2.set_ylabel('Silhouette Coefficient')
    
    # ax1.legend(loc='upper right')
    # ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = [Patch(color='b', alpha=1, label='Ward Hierarchical Cluster'), Patch(color='r', alpha=1, label='GMM (KMeans++ Init) Cluster')]
    ax2.legend(handles=handles, loc='upper right')

    plt.title('Ward Hierarchical Cluster VS GMM Cluster')
    if show:
        plt.show()
    else:
        plt.savefig(f'Ward_vs_GMM_compare.png')
# compareWardAndGMM(8914, True)
# compareWardAndGMM(8914, False)

# 下面代码没有在正式部分使用
def plotGMMcluster():
    '''绘制高斯混合聚类结果'''
    X = utils.readCSV()
    y, model = gmm.GMMcluster(X)
    plotClusterResults(X, y, 'GMM')
    model.print_params()
    plt.show()
# plotGMMcluster()

# @utils.print_exec_time
def checkSilhouette(p, labels, score):
    '''检验计算正确性，设算出来系数是score，检验其是否正确\n
    未在正式代码使用，只用于测试'''
    raise ValueError("Abandoned")
    # from sklearn.metrics import silhouette_score
    # from sklearn.metrics import silhouette_samples
    # answer = silhouette_score(p, labels)
    # print(silhouette_samples(p, labels))
    # assert abs(answer - score) < 1e-6, f'answer: {answer}, score: {score}'
    
def testSilhouette1():
    '''检查正确性，未在正式代码使用，只用于测试 \n 
    检测轮廓系数的计算'''
    raise ValueError("Abandoned")
    # p = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=3, random_state=0)
    # labels = kmeans.fit_predict(p)
    # print(labels)
    # dis = utils.getDisMatrix(p)
    # score = calcSilhouette(dis, labels)
    # checkSilhouette(p, labels, score)
# testSilhouette1()
    
def testSilhouette2():
    '''检查正确性，未在正式代码使用，只用于测试 \n 检测轮廓系数的计算'''
    raise ValueError("Abandoned")
    # p = np.random.rand(5000, 2)
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=25, random_state=0)
    # labels = kmeans.fit_predict(p)
    # dis = utils.getDisMatrix(p)
    # score = calcSilhouette(dis, labels)
    # checkSilhouette(p, labels, score)
# testSilhouette2()

def checkGMM():
    '''检查高斯混合聚类结果，未在正式代码使用，只用于测试'''
    raise ValueError("Abandoned")
    # X = utils.readCSV()
    # from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=15, random_state=0)
    # X0 = utils.z_score(X)
    # y = gmm.fit_predict(X0)
    # plotClusterResults(X, y, 'GMM')
    # plt.show()
    # for i in range(gmm.n_components):
    #     print(i + 1)
    #     print(gmm.weights_[i])
    #     print(gmm.means_[i])
    #     print(gmm.covariances_[i])
# checkGMM()

def checkKMeans(strategy, seed=50):
    '''检查KMeans聚类结果，未在正式代码使用，只用于测试'''
    raise ValueError("Abandoned")
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # X = utils.readCSV()
    # X0 = utils.z_score(X)
    # model = gmm.KMeans(15, seed, strategy)
    # model.fit(X0)
    # y = model.predict(X0)
    # plt.subplot(1,2,1)
    # plotClusterResults(X0, y, 'KMeans '+strategy.title())
    # c = model.centroids
    # plt.scatter(c[:, 0], c[:, 1], c='black', marker='x')

    # from sklearn.cluster import KMeans
    # model2 = KMeans(15, random_state=seed)
    # model2.fit(X0)
    # y2 = model2.predict(X0)
    # plt.subplot(1,2,2)
    # plotClusterResults(X0, y2, 'KMeans '+strategy.title())
    # c = model2.cluster_centers_
    # plt.scatter(c[:, 0], c[:, 1], c='black', marker='x')
    # plt.show()
# checkKMeans('random')
# checkKMeans('kmeans++', 996)

@utils.print_exec_time
def checkKMedoids():
    '''检查KMedoids聚类结果，未在正式代码使用，只用于测试'''
    X = utils.readCSV()
    X0 = utils.z_score(X)
    model = gmm.KMedoids(15, 50)
    model.fit(X0)
    y = model.predict(X0)
    plotClusterResults(X, y, 'KMedoids')
    plt.show()
# checkKMedoids()

def checkSSE():
    '''用 calcSSEs 测试 calcSSE 的正确性，结果表明 calcSSE 的 SSE 计算正确'''
    with open('steps_ward.txt', 'r') as f:
        steps = eval(f.read())
    p = utils.readCSV()
    n = 5000
    dsu = disjointSet.DSU_SSE(n, p)
    for i in range(n - 15):
        u, v, _ = steps[i]
        dsu.merge(u, v)
    labels = disjointSet.getClasses(dsu)
    sse2 = calcSSE(p, labels, 15)
    sse1 = dsu.sse
    print(sse1, sse2) # 9054838502187.725 9054838502187.762
# checkSSE()

def test_gmm_save_and_load():
    '''检测GMM模型的可复现性，训练后保存模型参数，结果检查正确'''
    p = utils.readCSV()
    label, model = gmm.GMMcluster(p)
    plotClusterResults(p, label, 'GMM1')
    plt.show()
    model.save_params('gmm.pkl')
    # model.print_params()
    model2 = gmm.GMM()
    model2.load_params('gmm.pkl.npz')
    # model2.print_params()
    label2 = model2.predict(utils.z_score(p))
    plotClusterResults(p, label2, 'GMM2')
    plt.show()
# test_gmm_save_and_load()