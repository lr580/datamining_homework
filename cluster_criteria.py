# 测试聚类结果，以及可视化相关代码
from disjointSet import DSU_SSE # 手写并查集
import cluster # 手写层次聚类
import utils # 手写辅助函数
import matplotlib.pyplot as plt 
import numpy as np
# @utils.print_exec_time # 0.02s
def calcSSE(n, steps, p):
    '''给定聚类过程steps和点集p，计算聚成k∈[1,n]类各自的SSE，返回0-indexed的SSE列表 \n 复杂度 O(nα)≈O(n)'''
    dsu = DSU_SSE(n, p)
    sse = [0] * n
    for i in range(n - 1):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        sse[-(i+2)] = dsu.sse
    return sse

def plotSSE(sse, type_, logScale=False):
    plt.plot(np.arange(1, len(sse) + 1), sse, marker='o')
    if logScale:
        plt.yscale('log')
    plt.title(f'SSE {type_}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.grid()

def plotSSEs():
    '''绘图展示四种聚类的SSE'''
    p = utils.readCSV()
    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i, type_ in enumerate(cluster.ALL_TYPES):
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        sse = calcSSE(p.shape[0], steps, p)
        plt.subplot(2,2,i+1)
        plotSSE(sse[:25], type_)
    plt.tight_layout()
    # plt.show()
    plt.savefig('SSE_partial.png')
plotSSEs()