from typing import List
from disjointSet import DSU
def minCluster(a:List[List[float]], k:int):
    '''输入n阶方阵a代表距离矩阵，最终聚成k类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v)表示在第i步把点u,v相连，其中u,v∈[0,n) \n
    算法复杂度：O(n^2logn + n^2α) = O(n^2logn) 取排序复杂度，其中 α 是反阿克曼函数；可以考虑用基数排序进一步优化排序'''
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)] # 对称矩阵只需要三角
    dsu = DSU(n)
    steps = []
    for dis, u, v in sorted(d):
        if dsu.merge(u, v):
            # print(dis, u+1, v+1)
            steps.append((u, v))
            if len(steps) == n - k:
                break
        # else: print('skip', dis, u+1, v+1)
    maps = dict() # 把DSU的[0,n)分类映射到[0,k)
    for i in range(n):
        if dsu.findFa(i) == i:
            maps[dsu.findFa(i)] = len(maps)
    clusters = [maps[dsu.fa[i]] for i in range(n)]
    return clusters, steps

# 测试用例：PPT例子
'''
testcase = [
    [0, 0.24, 0.22, 0.37, 0.34, 0.23],
    [0.24, 0, 0.15, 0.20, 0.14, 0.25],
    [0.22, 0.15, 0, 0.15, 0.28, 0.11],
    [0.37, 0.20, 0.15, 0, 0.29, 0.22],
    [0.34, 0.14, 0.28, 0.29, 0, 0.39],
    [0.23, 0.25, 0.11, 0.22, 0.39, 0]
]
clusters, steps = minCluster(testcase, 1)
print(clusters)
print(steps)
'''