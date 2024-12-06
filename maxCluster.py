from typing import List
from disjointSet import DSU_max
def maxCluster(a:List[List[float]], k:int):
    '''输入n阶方阵a代表距离矩阵，最终聚成k类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v)表示在第i步把点u,v相连，其中u,v∈[0,n) \n
    算法复杂度：O(n^2logn + n^2α) = O(n^2logn) 取排序复杂度，其中 α 是反阿克曼函数；可以考虑用基数排序进一步优化排序 \n
    注意最多合并 n 次 DSU.max，每次合并 O(n)，故复杂度不变'''
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)]
    dsu = DSU_max(n, a)
    steps = []
    for dis, u, v in sorted(d):
        fu, fv = dsu.findFa(u), dsu.findFa(v)
        print(dis, u+1, v+1, dsu.max[fu][fv], fu+1, fv+1)
        if dsu.max[fu][fv] == dis:
            dsu.merge(fu, fv)
            steps.append((u+1, v+1))
            print(dis,u+1,v+1)
            if len(steps) == n - k:
                break
    maps = dict() # 把DSU的[0,n)分类映射到[0,k)
    for i in range(n):
        if dsu.findFa(i) == i:
            maps[dsu.findFa(i)] = len(maps)
    clusters = [maps[dsu.fa[i]] for i in range(n)]
    return clusters, steps

# 测试用例：PPT例子

testcase = [
    [0, 0.24, 0.22, 0.37, 0.34, 0.23],
    [0.24, 0, 0.15, 0.20, 0.14, 0.25],
    [0.22, 0.15, 0, 0.15, 0.28, 0.11],
    [0.37, 0.20, 0.15, 0, 0.29, 0.22],
    [0.34, 0.14, 0.28, 0.29, 0, 0.39],
    [0.23, 0.25, 0.11, 0.22, 0.39, 0]
]
clusters, steps = maxCluster(testcase, 1)
print(clusters)
print(steps)

