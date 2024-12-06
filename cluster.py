from typing import List
from disjointSet import DSU, DSU_max, DSU_ele, getClasses
from heap import HeapMap
import copy, utils
def minCluster(a:List[List[float]], k:int):
    '''最小层次聚类 Hierarchical Clustering: MIN\n
    输入n阶方阵a代表距离矩阵，最终聚成k类 \n
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
    '''最大层次聚类 Hierarchical Clustering: MAX\n
    输入、返回值、复杂度描述同 minCluster() \n 
    注意最多合并 n 次，每次合并 O(n)，故复杂度不变'''
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)]
    dsu = DSU(n)
    e = copy.deepcopy(a)
    steps = []
    for dis, u, v in sorted(d):
        fv, fu = sorted([dsu.findFa(u), dsu.findFa(v)])
        if e[fu][fv] == dis: # fu 合并到 fv，删除 fu
            for i in range(n): # 最多合并 O(n) 次，每次 O(n) 复杂度
                maxe = max(e[fu][i], e[fv][i])
                e[fv][i] = e[i][fv] = maxe
                e[fu][i] = e[i][fu] = 0 # 方便 debug，其实可以不删
            # utils.print2Darray(e)
            dsu.merge(fu, fv)
            steps.append((u, v))
            if len(steps) == n - k:
                break
    return getClasses(dsu), steps

def avgCluster(a:List[List[float]], k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average \n
    输入、返回值、复杂度描述同 maxCluster() \n 
    复杂度分析：每次合并时，会修改 O(n) 个点对，永久删除 O(n) 个点对，且单次增删改是 logn 复杂度，故总复杂度为 O(n^2logn)'''
    n = len(a)
    steps = []
    dsu = DSU_ele(n)
    e = HeapMap()
    for u in range(n):
        for v in range(u+1, n):
            # 均值信息 = (均值，点数，总和)
            e.add((u, v), (a[u][v], 1, a[u][v]))
    remain = set(i for i in range(n)) # 剩余类
    # cnt = 0 # 调试用，计算复杂度
    def pair(u,v): # 转换为 HeapMap 的键
        return tuple(sorted([u, v]))
    while len(remain) > k:
        (u, v), _ = e.getMin()
        fv, fu = sorted([dsu.findFa(u), dsu.findFa(v)])
        if fv != fu: # fu 合并到 fv，删除 fu
            remain.remove(fu)
            # 更新其他部分跟合并后新部分的组间平均值
            for t in remain - {fv}:
                k1 = pair(fu, t)
                k2 = pair(fv, t)
                _, n1, s1 = e.k2v[k1]
                _, n2, s2 = e.k2v[k2]
                newVal = ((s1+s2)/(n1+n2), n1+n2, s1+s2)
                e.modify(pair(fv, t), newVal)
                # cnt += 1

            for u1 in dsu.ele[fu]:
                for v1 in dsu.ele[fv]: # 现在组内的边都可以删了
                    e.erase(pair(u1, v1))
                    # cnt += 1
                for t in remain- {fv}: # fu的也可以删了
                    e.erase(pair(u1, t))
                    # cnt += 1
            
            dsu.merge(fv, fu)
            steps.append((u, v))
    # print(cnt)
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
utils.print2Darray(testcase) # 测试用例输出
clusters, steps = minCluster(testcase, 1)
assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
assert check_correct(testcase, steps, 'single')
print(steps)
clusters, steps = maxCluster(testcase, 1)
assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
print(steps)
assert check_correct(testcase, steps, 'complete')
assert check_correct(testcase, steps, 'single') == False

clusters, steps = avgCluster(testcase, 1)
assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
assert check_correct(testcase, steps, 'average')
print(steps)

# 复杂度测试验证 cnt
'''
for n in (10,100,1000):
    a = [[0 for i in range(n)] for j in range(n)]
    print(n, end=' : ')
    avgCluster(a, 1)
'''
'''实验结果：表明确实大约 n^2 级别的操作次数
10 : 123
100 : 16696
1000 : 1739697'''