from typing import List
from disjointSet import DSU, DSU_ele, DSU_avg, DSU_cluster, getClasses # 手写实现的并查集数据结构
from heap import HeapMap, TreeMap # 手写实现的可删堆数据结构
import utils # 手写辅助函数
import numpy as np # 加速运算
import heapq 

# 计时装饰器，可以取消，主要用于调试
@utils.print_exec_time
def cluster(p:List[List[float]], type_:str, k:int=1):
    '''层次聚类 \n
    输入：p(n,2)的numpy数组代表n个欧式平面点 \n
    k 代表最终要聚成几个类 \n
    type_ 代表聚类方式，可选：'single', 'complete', 'average', 'ward' \n
    分别代表最小、最大、组平均、ward 层次聚类 \n
    返回值：1. 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    2. 长为n-k的二元组数组，第i个元素(u,v,w)表示在第i步把点u,v相连，其中u,v∈[0,n),w是合并前u,v的距离 \n
    时间复杂度 O(n^2logn)，空间复杂度 O(n^2) \n
    即 O(n^2logn + n^2α) = O(n^2logn) 取排序复杂度，其中 α 是反阿克曼函数，有 α < logn
    '''
    if type_ == 'single' or type_ == 'min':
        # a = utils.DisMatrix(p) # 15s
        # a = utils.getDisMatrix(p) # 15.8s
        return minCluster(p, k)
    elif type_ == 'complete' or type_ == 'max':
        return maxCluster(p, k)
    elif type_ == 'average' or type_ == 'avg':
        # return avgCluster(p, k)
        # return averageCluster_(p, k)
        return averageCluster(p, k)
    else: # ward
        return wardCluster(p, k)
    
def getOrderedDistList(p, isPointsets=True):
    '''给定点集p[n][2]或距离矩阵p[n][n]，(isPointSets区分) \n
    返回升序排列的 n*(n-1)/2 个距离对 (dis, u, v) \n
    其中 dis 是两点欧氏距离，u,v 是点的索引，u<v，0-indexed \n
    使用 numpy 探索了充分的优化，复杂度 O(n^2logn) 且常数较小'''
    # 下面是排序距离部分
    '''朴素实现 
    n = len(a)
    d = [(a[u][v], u, v) for u in range(n) for v in range(u+1,n)] # 对称矩阵只需要三角; # 2.45s
    d = sorted(d) #11.83s'''

    #numpy优化上面注释的朴素实现
    if isPointsets:
        a = utils.getDisMatrix(p)
    else:
        a = p
    # timer = utils.Timer()
    ij_pair = np.triu_indices(p.shape[0], k=1)
    dtype = [('dis', 'float'), ('u', 'int'), ('v', 'int')]
    vals = a[ij_pair]
    d = np.empty(len(vals), dtype=dtype)
    d['dis'] = vals
    d['u'] = ij_pair[0]
    d['v'] = ij_pair[1]
    # timer() # 构造数组 0.28s
    return d[np.argsort(d['dis'])] # 排序: 1.47s
    # 若：d.sort(order='dis') 则排序：8s
    # timer()

def minCluster(p, k:int):
    '''最小层次聚类 Hierarchical Clustering: MIN\n
    输入、返回值描述同 cluster() '''
    n = p.shape[0]
    dsu = DSU(n)
    steps = []
    cnt = 0
    for dis, u, v in getOrderedDistList(p):
        cnt += 1
        if dsu.merge(u, v):
            steps.append((u, v, dis))
            if len(steps) == n - k:
                break
    print(cnt)
    return getClasses(dsu), steps

def maxCluster(p, k:int):
    '''最大层次聚类 Hierarchical Clustering: MAX\n
    输入、返回值、复杂度描述同 minCluster() \n 
    注意最多合并 n 次，每次合并 O(n)，故复杂度不变'''
    n = p.shape[0]
    dsu = DSU(n)
    e = utils.getDisMatrix(p)
    steps = []
    op = getOrderedDistList(e, False)
    # cnt = 0 # 测试，统计执行次数，判断 break 的影响
    for dis, u, v in op:
        # cnt += 1
        fv, fu = sorted([dsu.findFa(u), dsu.findFa(v)])
        if e[fu,fv] == dis: # fu 合并到 fv，删除 fu
            ''' 朴素代码
            for i in range(n): # 最多合并 O(n) 次，每次 O(n) 复杂度
                maxe = max(e[fu][i], e[fv][i])
                e[fv][i] = e[i][fv] = maxe
                e[fu][i] = e[i][fu] = 0 # 方便 debug 输出，其实可以不删
            '''
            # numpy 优化，能快 20s
            maxe = np.maximum(e[fu,:],e[fv,:])
            e[fv,:] = maxe
            e[:,fv] = maxe
            # utils.print2Darray(e)
            dsu.merge(fu, fv)
            steps.append((u, v, dis))
            if len(steps) == n - k:
                break
    # print(cnt)
    return getClasses(dsu), steps

def averageCluster(p, k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average  \n 未清除优化过程的注释'''
    # memoryer = utils.MemoryTracker() # n=5000 97.7520s 2113.75MB
    n = p.shape[0]
    a = utils.getDisMatrix(p)
    q = getOrderedDistList(a, False).tolist() # 最小堆
    # heapq.heapify(q) # 本身有序
    steps = []
    # dsu = DSU_avg(n, n, a)
    dsu = DSU_cluster(n, n, a, DSU_cluster.add)
    alive = {i for i in range(n)} # 还活着的每组的最大点
    while q:
        dis, u, v = heapq.heappop(q)
        if u in alive and v in alive and dsu.merge(u, v):
            # timer = utils.Timer()
            steps.append((u, v, dis))
            alive.remove(u)
            alive.remove(v)
            alive.add(dsu.top)
            for i in alive:
                fu, fv = pair(dsu.findFa(i), dsu.findFa(dsu.top))
                dis = dsu.e[fu, fv] / (dsu.siz[fu] * dsu.siz[fv])
                heapq.heappush(q, (dis, *pair(i, dsu.top)))
            # timer()
            if len(steps) == n - k:
                break
    return getClasses(dsu)[:n], steps

def pair(u,v): # 辅助函数，转换为 HeapMap 的键，即排序 u,v
    return tuple(sorted([u, v]))

def avgCluster(p, k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average \n
    输入、返回值、复杂度描述同 maxCluster() \n 
    复杂度分析：每次合并时，会修改 O(n) 个点对，永久删除 O(n) 个点对，且单次增删改是 logn 复杂度，故总复杂度为 O(n^2logn) \n 效率太低，已废置'''
    # memoryer = utils.MemoryTracker()
    # timer = utils.Timer()
    n = p.shape[0]
    a = utils.getDisMatrix(p)
    steps = []
    dsu = DSU_ele(n)
    e = TreeMap(n)
    cnt = 0
    for u in range(n):
        for v in range(u+1, n): # 0.02s each (n=5000), total 100s
            # 均值信息 = (点数，总和)，其中 均值 = 总和/点数
            e.add(u, v, 1, a[u][v])
    remain = set(i for i in range(n)) # 剩余类
    # cnt = 0 # 调试用，计算复杂度
    step = 0 # 展示进度
    # timer() # 94.61s(n=5000)
    while len(remain) > k:
        u, v, dis = e.getMin()
        fv, fu = sorted([dsu.findFa(u), dsu.findFa(v)])
        if fv != fu: # fu 合并到 fv，删除 fu
            remain.remove(fu)
            # 更新其他部分跟合并后新部分的组间平均值
            for t in remain - {fv}:
                n1, s1 = e.get(*pair(fu, t))
                n2, s2 = e.get(*pair(fv, t))
                e.modify(*pair(fv, t), n1+n2, s1+s2)
                # cnt += 1

            for u1 in dsu.ele[fu]:
                for v1 in dsu.ele[fv]: # 现在组内的边都可以删了
                    e.erase(*pair(u1, v1))
                    # cnt += 1
                for t in remain - {fv}: # fu的也可以删了
                    e.erase(*pair(u1, t))
                    # cnt += 1
            
            dsu.merge(fv, fu)
            steps.append((u, v, dis))
            step += 1
            # print(step, end = ' ')
            # memoryer() # 1988.21MB(n=5000)
    # print(cnt)
    return getClasses(dsu), steps

def averageCluster_(p, k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average  \n 未清除优化过程的注释'''
    # memoryer = utils.MemoryTracker() # n=5000 97.7520s 2113.75MB
    timer = utils.Timer()
    n = p.shape[0]
    a = utils.getDisMatrix(p)
    q = getOrderedDistList(a, False).tolist() # 最小堆
    # heapq.heapify(q) # 本身有序
    steps = []
    dsu = DSU_cluster(n, n, a, DSU_cluster.add)
    # alive = {i for i in range(n)} # 还活着的每组的最大点
    alive = np.array([1] * n + [0] * n)
    timer()
    cnt = 0
    while q:
        dis, u, v = heapq.heappop(q)
        cnt += 1
        # if u in alive and v in alive and dsu.merge(u, v):
        if alive[u] and alive[v] and dsu.merge(u, v):
            steps.append((u, v, dis))
            # alive.remove(u)
            # alive.remove(v)
            # alive.add(dsu.top)
            alive[u] = 0
            alive[v] = 0
            alive[dsu.top] = 1
            
            
            # for i in np.where(alive == 1)[0]:
            '''for i in alive:
                fu, fv = pair(dsu.findFa(i), dsu.findFa(dsu.top))
                dis = dsu.e[fu, fv] / (dsu.siz[fu] * dsu.siz[fv])
                heapq.heappush(q, (dis, *pair(i, dsu.top)))'''
            
            # 一些优化的尝试：
            alive_i = np.where(alive == 1)[0]
            # alive_i = np.array(list(alive))
            fu_list = np.array([dsu.findFa(i) for i in alive_i])
            fv = dsu.findFa(dsu.top)
            # timer()
            dis = dsu.e[fu_list, fv] / (dsu.siz[fu_list] * dsu.siz[fv])
            # timer()
            # for i, dis in zip(alive_i, dis): # 102s
            #     u, v = pair(i, dsu.top)
            #     heapq.heappush(q, (dis, u, v))
            new_eles = [(dis, i, dsu.top) for i, dis in zip(alive_i, dis)]
            # new_eles = [(dis, *pair(i, dsu.top)) for i, dis in zip(alive_i, dis)]
            # new_keys = np.array([np.sort([i, dsu.top]) for i in alive_i]) # 批量pair
            # new_eles = np.column_stack((dis, new_keys))
            # new_eles = np.column_stack((dis, alive_i, np.full_like(alive_i, dsu.top)))
            # timer()
            for e in new_eles: # 104s
                heapq.heappush(q, (e[0],int(e[1]),int(e[2]))) 
            # timer()
            # print()
            if len(steps) % 100 == 0:
                timer()
            if len(steps) == n - k:
              break
    print(cnt) # 17491237
    return getClasses(dsu)[:n], steps

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
                for t in remain - {fv}: 
                    e.erase(pair(u1, t))

            dsu.merge(fv, fu)
            for t in remain - {fv}:
                e.modify(pair(fv, t), dist(fv, t))
            steps.append((u, v))
    return getClasses(dsu), steps
    

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
    # ('complete',  'single', 'average', 'ward')
    for type_ in ('average', ): # , 'average', 'ward'
        clusters, steps = cluster(data, type_, 1)
        # assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
        # assert check_correct2(data, steps, type_)

    ''' 测试报告：
    minCluster: 对 n=5000 用时3.2s，符合预期 O(n^2logn) 的效率
    '''
# utils.chcp()
testcase2(utils.reconstruct_points(utils.getPPTsampleMatrix()))
# testcase2(utils.readCSV())
