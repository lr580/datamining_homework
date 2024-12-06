from typing import List
from disjointSet import DSU, DSU_max, getClasses
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
    return getClasses(dsu), steps

def maxCluster(a:List[List[float]], k:int):
    '''输入n阶方阵a代表距离矩阵，最终聚成k类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v)表示在第i步把点u,v相连，其中u,v∈[0,n) \n
    算法复杂度：O(n^2logn + n^2α) = O(n^2logn) \n
    注意最多合并 n 次 DSU.max，每次合并 O(n)，故复杂度不变'''
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)]
    dsu = DSU_max(n, a)
    steps = []
    for dis, u, v in sorted(d):
        fu, fv = dsu.findFa(u), dsu.findFa(v)
        mx = max(dsu.max[fu][fv], dsu.max[fv][fu]) # 并查集只更新合并的那一部分对其他点的距离，没有更新其他点对合并的距离，所以是不对称的
        # print(dis, u+1, v+1, mx, fu+1, fv+1)
        if mx == dis:
            dsu.merge(fu, fv)
            steps.append((u, v))
            if len(steps) == n - k:
                break
    return getClasses(dsu), steps


def check_correct(a, steps, type_:str):
    '''将手写代码与库函数对比，以验证正确性 \n
    调库只用来测试正确性，没有用来后续正式使用'''
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    ans = linkage(squareform(a), type_)
    n = len(a)
    dsu1, dsu2 = DSU(n), DSU(n*2)
    for i in range(len(steps)):
        u1, v1 = steps[i][0], steps[i][1]
        dsu1.merge(v1, u1)
        u2, v2 = int(ans[i][0]), int(ans[i][1])
        u2, v2 = sorted([dsu2.findFa(u2), dsu2.findFa(v2)])
        dsu2.merge(v2, u2)
        dsu2.merge(n+i, u2)
        fu1, fv1 = dsu1.findFa(u1), dsu1.findFa(v1)
        fu2, fv2 = dsu2.findFa(u2), dsu2.findFa(v2)
        if sorted([fu1, fv1]) != sorted([fu2, fv2]):
            return False
    return True

# 测试用例：PPT例子
testcase = [
    [0, 0.24, 0.22, 0.37, 0.34, 0.23],
    [0.24, 0, 0.15, 0.20, 0.14, 0.25],
    [0.22, 0.15, 0, 0.15, 0.28, 0.11],
    [0.37, 0.20, 0.15, 0, 0.29, 0.22],
    [0.34, 0.14, 0.28, 0.29, 0, 0.39],
    [0.23, 0.25, 0.11, 0.22, 0.39, 0]
]
clusters, steps = minCluster(testcase, 1)
assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
assert check_correct(testcase, steps, 'single')
print(steps)
clusters, steps = maxCluster(testcase, 1)
assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
assert check_correct(testcase, steps, 'complete')
assert check_correct(testcase, steps, 'single') == False
print(steps)


