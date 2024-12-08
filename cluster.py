from typing import List
from disjointSet import DSU, DSU_ele, DSU_avg, getClasses # 手写实现的并查集数据结构
from heap import HeapMap # 手写实现的可删堆数据结构
import copy
import utils # 手写辅助函数
import numpy as np # 加速运算

def minCluster(a, k:int):
    '''最小层次聚类 Hierarchical Clustering: MIN\n
    输入n阶方阵a代表距离矩阵，最终聚成k类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v,w)表示在第i步把点u,v相连，其中u,v∈[0,n),w是合并前u,v的距离 \n
    算法复杂度：O(n^2logn + n^2α) = O(n^2logn) 取排序复杂度，其中 α 是反阿克曼函数，有 α < logn；可以考虑用基数排序进一步优化排序'''
    

    # 下面是排序距离部分
    '''朴素实现 
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)] # 对称矩阵只需要三角; # 2.45s
    d = sorted(d) #11.83s'''
    #numpy优化上面注释的朴素实现
    n = a.shape[0]
    # timer = utils.Timer()
    ij_pair = np.triu_indices(n, k=1)
    dtype = [('dis', 'float'), ('u', 'int'), ('v', 'int')]
    vals = a[ij_pair]
    d = np.empty(len(vals), dtype=dtype)
    d['dis'] = vals
    d['u'] = ij_pair[0]
    d['v'] = ij_pair[1]
    # timer() # 构造数组 0.28s
    d = d[np.argsort(d['dis'])] # 排序: 1.47s
    # 若：d.sort(order='dis') 则排序：8s
    # timer()

    dsu = DSU(n)
    steps = []
    for dis, u, v in d:
        if dsu.merge(u, v):
            steps.append((u, v, dis))
            if len(steps) == n - k:
                break

    return getClasses(dsu), steps

def maxCluster(a, k:int):
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
                e[fu][i] = e[i][fu] = 0 # 方便 debug 输出，其实可以不删
            # utils.print2Darray(e)
            dsu.merge(fu, fv)
            steps.append((u, v))
            if len(steps) == n - k:
                break
    return getClasses(dsu), steps

def pair(u,v): # 辅助函数，转换为 HeapMap 的键，即排序 u,v
    return tuple(sorted([u, v]))

def avgCluster(a, k:int):
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

def wardCluster(p, k:int):
    '''Ward 聚类 Hierarchical Clustering: Ward \n
    输入：(n,2)的数组代表n个欧式平面点 \n
    返回值、复杂度描述同 maxCluster() \n 
    距离公式：两重心的距离的平方乘以点数的调和平均的二倍
    '''
    n = len(p)
    steps = []
    dsu = DSU_avg(n, p)
    def dist(fu, fv): # 求两个簇的ward距离
        m1, m2 = dsu.avg(fu), dsu.avg(fv)
        n1, n2 = len(dsu.ele[fu]), len(dsu.ele[fv])
        dis = ((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
        return dis * 2 * n1 * n2 / (n1 + n2)
    e = HeapMap()
    for u in range(n):
        for v in range(u+1, n):
            e.add((u, v), dist(u, v))
    remain = set(i for i in range(n))
    while len(remain) > k:
        (u, v), dis = e.getMin()
        fv, fu = sorted([dsu.findFa(u), dsu.findFa(v)])
        if fv != fu: # fu 合并到 fv，删除 fu
            # print(u, v, dis**0.5) # 调试用，验证正确性
            remain.remove(fu)
            for u1 in dsu.ele[fu]:
                for v1 in dsu.ele[fv]: 
                    e.erase(pair(u1, v1))
                for t in remain- {fv}: 
                    e.erase(pair(u1, t))

            dsu.merge(fv, fu)
            for t in remain - {fv}:
                e.modify(pair(fv, t), dist(fv, t))
            steps.append((u, v))
    return getClasses(dsu), steps

# 计时装饰器，可以取消，主要用于调试
@utils.print_exec_time
def cluster(p:List[List[float]], type_:str, k:int=1):
    '''层次聚类 \n
    输入：p(n,2)的数组代表n个欧式平面点 \n
    k 代表最终要聚成几个类 \n
    type_ 代表聚类方式，可选：'single', 'complete', 'average', 'ward' \n
    分别代表最小、最大、组平均、ward 层次聚类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v,w)表示在第i步把点u,v相连，其中u,v∈[0,n),w是合并前u,v的距离 \n
    时间复杂度 O(n^2logn)，空间复杂度 O(n^2)
    '''
    if type_ == 'single' or type_ == 'min':
        # a = utils.DisMatrix(p) # 15s
        a = utils.getDisMatrix(p) # 15.8s
        return minCluster(a, k)
    elif type_ == 'complete' or type_ == 'max':
        a = utils.getDisMatrix(p)
        return maxCluster(a, k)
    elif type_ == 'average' or type_ == 'avg':
        a = utils.getDisMatrix(p)
        return avgCluster(a, k)
    else: # ward
        return wardCluster(p, k)
    

def check_correct(a, steps, type_:str, matrix=True):
    '''将手写代码与库函数对比，以验证正确性 \n
    调库只用来测试正确性，没有用来后续正式使用 \n 
    该测试代码针对 79f8cd 版本前的代码，对修改后不适用'''
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    if matrix:
        ans = linkage(squareform(a), type_)
    else:
        ans = linkage(a, type_)

        # print(ans)  # 对比 wardCluster 的 u, v, dis**0.5 输出，一样
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

def check_correct2(a, steps, type_:str):
    '''考虑到边权相等时合并顺序任意，这里对点集距离和steps距离进行对比'''
    from scipy.cluster.hierarchy import linkage
    ans = linkage(a, type_)
    for i in range(len(steps)):
        w1 = steps[i][-1]
        w2 = ans[i][2]
        if abs(w1-w2) > 1e-6:
            return False
    return True

def testcase1():
    '''使用PPT例子对聚类进行测试，验证正确性 \n
    该测试代码针对 79f8cd 版本前的代码，对修改后不适用'''
    # 测试用例：PPT例子
    testcase = utils.getPPTsampleMatrix()
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

    testcase_pointwise = utils.reconstruct_points(testcase).tolist()
    clusters, steps = wardCluster(testcase_pointwise, 1)
    assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
    assert check_correct(testcase_pointwise, steps, 'ward', False)
    print(steps)
# testcase1()

def testcase2(data):
    '''直接使用作业数据测试，并验证正确性和效率优化'''
    for type_ in ('single', 'complete', 'average', 'ward'):
        clusters, steps = cluster(data, type_, 1)
        assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
        assert check_correct2(data, steps, type_)
        break
# testcase2(utils.readCSV())
utils.chcp()
testcase2(utils.reconstruct_points(utils.getPPTsampleMatrix()))
testcase2(utils.readCSV())