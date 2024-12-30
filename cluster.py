from typing import List # 代码函数参数提示
from disjointSet import DSU, DSU_average, DSU_ward, getClasses, DSU_hard, DSU_top # 手写实现的并查集数据结构
import utils # 手写辅助函数
import numpy as np # 加速运算
import heapq # 优先级队列(最小堆)
import os # 文件操作

ALL_TYPES = ('single', 'complete','average', 'ward') 
'''所有四种层次聚类的名字'''

# 手写的计时装饰器，可以取消，主要用于调试 (运行时间根据电脑性能不同存在差异)
# @utils.print_exec_time
def cluster(p:List[List[float]], type_:str, k:int=1):
    '''层次聚类 \n
    输入：p 是(n,2)的numpy数组代表n个欧式平面点 \n
    k 代表最终要聚成几个类 \n
    type_ 代表聚类方式，可选：'single', 'complete', 'average', 'ward' \n
    分别代表最小、最大、组平均、ward 层次聚类 \n
    
    返回值：clusters 和 steps 的元祖 \n
    clusters: 长为n的整数数组表示每个点所属的类取值∈[0,k) \n
    steps: 长为n-k的二元组数组，第i个元素(u,v,w)表示在第i步把点u,v相连，其中u,v∈[0,n),w是合并前u,v的距离 \n
    
    时间复杂度 O(n^2logn)，空间复杂度 O(n^2) \n
    即 O(n^2logn + n^2α) = O(n^2logn) 取排序复杂度，其中 α 是反阿克曼函数，有 α < logn \n
    
    返回值解释示例：(以课堂lec4 PPT 58 页例子为例) (这里下标0开始，课件1开始) \n
    假设 steps=[[2,5,0.11],[1,4,0.14],[1,2,0.15],[2,3,0.15],[0,2,0.22]] \n
    表示点2与点5所在的类合并，合并前距离为0.11，此时 0,1,(2,5),3,4 \n
    然后点1与点4所在的类合并，合并前距离为0.14，此时 0,(1,4),(2,5),3 \n
    然后点1与点2所在的类合并，合并前距离为0.15，此时 0,(1,2,4,5),3 \n
    然后点2与点3所在的类合并，合并前距离为0.15，此时 0,(1,2,3,4,5) \n
    最后点0与点2所在的类合并，合并前距离为0.22，此时 (0,1,2,3,4,5) \n
    clusters=[0,0,0,0,0,0] 表示每个点都在类别0中 (假设k=1)'''
    if type_ == 'single': # 'min':
        return minCluster(p, k)
    elif type_ == 'complete': # 'max':
        return maxCluster(p, k)
    elif type_ == 'average': # 'avg':
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
    for dis, u, v in getOrderedDistList(p):
        if dsu.merge(u, v):
            steps.append((u, v, dis))
            if len(steps) == n - k:
                break
    return getClasses(dsu), steps

def maxCluster(p, k:int):
    '''最大层次聚类 Hierarchical Clustering: MAX\n
    输入、返回值、复杂度描述同 minCluster() \n 
    注意最多合并 n 次，每次合并 O(n)，故复杂度不变'''
    n = p.shape[0]
    dsu = DSU(n)
    e = utils.getDisMatrix(p)
    steps = []
    op = getOrderedDistList(e, False).tolist()
    # cnt = 0 # 测试，统计执行次数，判断 break 的影响
    # timer = utils.Timer()
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
            # if len(steps) % 100 == 99 or len(steps) >= 4900: timer() # 测试，统计执行时间分布
            # 4600 是经过多次调试效果较好的一个值，可以根据自己电脑自行再调整
            if len(steps) == 4600: # 仿照 average/wardCluster优化
                aliveP = [i for i in range(n) if dsu.findFa(i) == i]
                op = getOrderedDistList(p[aliveP]).tolist() # 从 24 秒优化到了 9秒，记得要tolist
            if len(steps) == n - k:
                break
    # print(cnt)
    return getClasses(dsu), steps

class HeapReconstructor:
    '''根据观察结果：>=4000时效率开始下降 \n
    因此考虑对需要删除元素大幅增加时，直接重构堆 \n
    观察验证对比代码：放在 cluster for/while 里 \n
    if len(steps) % 100 == 99 or len(steps) >= 4900: \n
        print(len(steps), len(q), len(alive)) \n
        timer()'''
    def __init__(self, q:List, alive, dsu, type_:str, SLEEP_LEN=100):
        self.q = q
        self.last_q_len = len(q) # 用于优化，当q长度下降时，进行q重构
        self.reconstructed = False # 是否已重构
        self.SLEEP_LEN = SLEEP_LEN # 间隔多少次检测激活尝试重构一次，根据数据大小和电脑性能可以调整(避免因为偶然长度下降提前重构)
        self.try_cnt = 0 # 当前尝试重构次数
        self.type_ = type_ # 'ward' 或 'avg'
        self.alive = alive
        self.dsu = dsu
        # self.timer = utils.Timer()
    def try_reset(self):
        '''尝试重构，每100次激活一次，激活后若满足重构标准进行重构，如果已经重构则不再重构'''
        if self.reconstructed:
            return
        self.try_cnt += 1
        if self.try_cnt % self.SLEEP_LEN == 0:
            # self.timer()
            if len(self.q) < self.last_q_len:
                self.reconstructed = True
                self.reset(self.alive, self.dsu)
            self.last_q_len = len(self.q)
    def reset(self, alive, dsu):
        '''重构堆'''
        # timer = utils.Timer()
        # print(f'堆重构：重构前元素{len(self.q)}, 剩余类：{len(alive)}') # 24330693, ࣺ600
        self.q.clear()
        nodes = list(alive)
        for i in range(len(nodes)):
            fu = dsu.findFa(nodes[i])
            for j in range(i+1, len(nodes)):
                fv = dsu.findFa(nodes[j])
                assert fu != fv
                if self.type_ == 'ward':
                    m1x = dsu.sx[fu] / dsu.siz[fu]
                    m1y = dsu.sy[fu] / dsu.siz[fu]
                    m2x = dsu.sx[fv] / dsu.siz[fv]
                    m2y = dsu.sy[fv] / dsu.siz[fv]
                    dis = ((m1x-m2x)**2 + (m1y-m2y)**2)
                    dis = dis * 2 * dsu.siz[fu] * dsu.siz[fv] / (dsu.siz[fu] + dsu.siz[fv])
                    self.q.append((dis, nodes[i], nodes[j]))
                elif self.type_ == 'avg':
                    dis = dsu.e[fu, fv] / (dsu.siz[fu] * dsu.siz[fv])
                    self.q.append((dis, nodes[i], nodes[j]))
        heapq.heapify(self.q)
        # print(f'重构完毕，堆元素：{len(self.q)}') # 179700
        # timer() # 1.72s

def averageCluster(p, k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average  \n 未清除优化过程的注释'''
    # memoryer = utils.MemoryTracker() # n=5000 97.7520s 2113.75MB
    n = p.shape[0]
    a = utils.getDisMatrix(p)
    q = getOrderedDistList(a, False).tolist() # 最小堆
    steps = []
    dsu = DSU_average(n, n, a)
    alive = {i for i in range(n)} # 还活着的每组的最大点
    q_resetter = HeapReconstructor(q, alive, dsu, 'avg')
    while q:
        dis, u, v = heapq.heappop(q)
        fu0, fv0 = dsu.findFa(u), dsu.findFa(v)
        if u in alive and v in alive and dsu.merge(u, v):
            # steps.append((u, v, dis)) # linakge 格式
            steps.append((fu0, fv0, dis)) # 注释描述格式
            alive.remove(u)
            alive.remove(v)
            alive.add(dsu.top)
            for i in alive:
                fu, fv = dsu.findFa(i), dsu.findFa(dsu.top)
                dis = dsu.e[fu, fv] / (dsu.siz[fu] * dsu.siz[fv])
                heapq.heappush(q, (dis, i, dsu.top))
            q_resetter.try_reset()
            if len(steps) == n - k:
                break
    # memoryer()
    return getClasses(dsu)[:n], steps

def wardCluster(p, k:int):
    '''Ward 聚类 Hierarchical Clustering: Ward \n
    距离公式：两重心的距离的平方乘以点数的调和平均的二倍'''
    n = p.shape[0]
    steps = []
    dsu = DSU_ward(n, n, p)
    q = getOrderedDistList(utils.getDisMatrixSquare(p), False) # 直接用平方计算，避免频繁的开方，提高速度
    q = q.tolist()
    # q = getOrderedDistList(p).tolist()
    alive = {i for i in range(n)}
    q_resetter = HeapReconstructor(q, alive, dsu, 'ward')
    # timer = utils.Timer()
    while q:
        dis, u, v = heapq.heappop(q)
        fu0, fv0 = dsu.findFa(u), dsu.findFa(v)
        if u in alive and v in alive and dsu.merge(u, v):
            # steps.append((u, v, dis**0.5))
            steps.append((fu0, fv0, dis**0.5))
            alive.remove(u)
            alive.remove(v)
            alive.add(dsu.top)
            ''' 优化前的朴素代码：
            for i in alive:
                fu, fv = dsu.findFa(i), dsu.findFa(dsu.top) # 147.2 / pair  # 166.8s
                m1, m2 = dsu.avg(fu), dsu.avg(fv)
                n1, n2 = dsu.siz[fu], dsu.siz[fv]
                dis = ((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
                dis = dis * 2 * n1 * n2 / (n1 + n2)
                heapq.heappush(q, (dis, i, dsu.top))'''
            # 优化后，代码速度快一倍
            fv = dsu.findFa(dsu.top)
            fu_list = np.array([dsu.findFa(i) for i in alive])
            m1x = dsu.sx[fu_list] / dsu.siz[fu_list]
            m1y = dsu.sy[fu_list] / dsu.siz[fu_list]
            m2x = dsu.sx[fv] / dsu.siz[fv]
            m2y = dsu.sy[fv] / dsu.siz[fv]
            dis = ((m1x-m2x)**2 + (m1y-m2y)**2)
            dis = dis * 2 * dsu.siz[fu_list] * dsu.siz[fv] / (dsu.siz[fu_list] + dsu.siz[fv])
            u_list = list(alive)
            for i in range(len(alive)):
                heapq.heappush(q, (dis[i], u_list[i], dsu.top))
            q_resetter.try_reset()
            if len(steps) == n - k:
                break
    return getClasses(dsu)[:n], steps

def generateAllClusterSteps(lazy=False):
    '''对 8gau.txt 数据集分别求四种聚类，得到的结果存储在根目录 steps_xxx.txt，其中 xxx=single, complete, average, ward；详见cluster函数注释和文档\n
    lazy表示是否文件存在时不进行计算直接使用结果，若lazy=True则如此'''
    p = utils.readCSV()
    for type_ in ALL_TYPES:
        path = f'steps_{type_}.txt'
        if os.path.exists(path) and lazy:
            continue
        print(f'正在计算{type_}聚类')
        labels, steps = cluster(p, type_)
        with open(path, 'w') as f:
            f.write(str(steps))
# generateAllClusterSteps(True)
# generateAllClusterSteps(False)

def ClusterFromStepsBuilder(n, steps, k=1,continous=True):
    '''对n个点，根据完全聚类(聚成1类)的步骤steps，构造一个聚成k类的步骤 \n
    使用生成器结构，方便一步一步地得到聚成n, n-1, ..., k+1, k 类结果 \n
    每次 yield 长为n的数组，表示每个点属于哪一类\n
    若 continous=True，则分类连续；单次 yield 复杂度 O(α+n)=O(n) \n
    否则分类值可能离散(但均摊复杂度更低) 均摊更快'''
    dsu = DSU_hard(n)
    for i in range(n-k):
        u, v, _ = steps[i]
        dsu.merge(u, v)
        yield getClasses(dsu) if continous else dsu.fa
# 测试代码
# p=utils.reconstruct_points(utils.getPPTsampleMatrix())
# _,steps=cluster(p, 'single', 1)
# builder=ClusterFromStepsBuilder(6,steps)
# for clusters in builder:
#     print(clusters)

def ClusterFromSteps(n, steps, k=1):
    '''对n个点，根据完全聚类(聚成1类)的步骤steps，聚类成k类，直接返回结果labels'''
    dsu = DSU(n)
    for i in range(n-k):
        u, v, _ = steps[i]
        dsu.merge(u, v)
    return getClasses(dsu)

def BuildPlottingSteps(steps):
    '''为了能绘制可视化图，将steps格式转化为绘图格式： \n
    n各点：[n-1][4] 数组：旧簇索引1 旧簇索引2 距离 新簇大小 \n
    时间复杂度 O(n)，基于并查集重构'''
    n = len(steps) + 1
    dsu = DSU_top(n*2)
    steps2 = np.zeros((n-1, 4), dtype=np.float64)
    top = n # 下一个类的编号
    for i in range(n-1):
        u, v, dis = steps[i]
        fu, fv = dsu.findFa(u), dsu.findFa(v)
        tu, tv = dsu.top[fu], dsu.top[fv]
        dsu.merge(fu, top)
        dsu.merge(fv, top)
        steps2[i] = np.array([tu, tv, dis, dsu.siz[dsu.findFa(top)]])
        top += 1
    return steps2



# 以下是测试正确性的测试用例代码，并未在正式代码中使用
def check_correct2(a, steps, type_:str):
    '''考虑到边权相等时合并顺序任意，这里对点集距离和steps距离进行对比 \n
    将手写代码与库函数对比，以验证正确性 \n
    调库只用来测试正确性，没有用来后续正式使用'''
    from scipy.cluster.hierarchy import linkage
    ans = linkage(a, type_)
    # print(ans)
    for i in range(len(steps)):
        w1 = steps[i][-1]
        w2 = ans[i][2]
        if abs(w1-w2) > 1e-6:
            return False
    return True

def testcase2(data):
    '''直接使用作业数据8gau.txt测试，并验证正确性和效率优化 \n
    测试报告： 对 n=5000 ,符合预期 O(n^2logn) 的效率 \n
    测试环境：13th Gen Intel(R) Core(TM) i7-13620H \n
    minCluster: 用时3.2s \n
    maxCluster: 用时10.8s \n
    averageCluster: 用时26.5s \n
    wardCluster: 用时14.0s'''
    for type_ in ALL_TYPES:
        clusters, steps = cluster(data, type_, 1)
        assert len(set(clusters)) == 1 and next(iter(clusters)) == 0
        assert check_correct2(data, steps, type_)
        # utils.print2Darray(steps)
        # 保存下来预处理，方便调试；也是作为结果输出展示
        # with open(f'steps_{type_}.txt', 'w') as f:
        #     f.write(str(steps))
# utils.chcp() # 若中文输出乱码，执行这个
# testcase2(utils.reconstruct_points(utils.getPPTsampleMatrix())) # 有损重构，跟课堂例子不完全一致
# testcase2(utils.readCSV())


# 往下是废置代码
def pair(u,v): # 辅助函数，转换为 HeapMap 的键，即排序 u,v
    return tuple(sorted([u, v]))

def avgCluster(p, k:int):
    '''平均层次聚类 Hierarchical Clustering: Group Average \n
    输入、返回值、复杂度描述同 maxCluster() \n 
    复杂度分析：每次合并时，会修改 O(n) 个点对，永久删除 O(n) 个点对，且单次增删改是 logn 复杂度，故总复杂度为 O(n^2logn) \n
    因效率过低已废置，重构见 averageCluster()'''
    from heap import TreeMap # 手写实现的可删堆数据结构
    from disjointSet import DSU_ele
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
    '''平均层次聚类 Hierarchical Clustering: Group Average  \n 
    未清除优化过程的注释 \n 
    因效率过低已废置，重构见 averageCluster()'''
    # memoryer = utils.MemoryTracker() # n=5000 97.7520s 2113.75MB
    from disjointSet import DSU_cluster
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

def wardCluster_(p, k:int):
    '''Ward 聚类 Hierarchical Clustering: Ward \n
    输入：(n,2)的数组代表n个欧式平面点 \n
    返回值、复杂度描述同 maxCluster() \n 
    距离公式：两重心的距离的平方乘以点数的调和平均的二倍 \n 
    因效率过低已废置，重构见 wardCluster()'''
    from heap import HeapMap # 红黑树+辅助结构
    from disjointSet import DSU_avg
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

def checkStepsReconstruction(steps, p, type_):
    '''检查重构绘图的steps是否满足绘图库函数标准 \n
    经过检验：实际上本质是一样的，差异是由于相等权重的顺序不一样导致的'''
    from scipy.cluster.hierarchy import linkage
    linked = linkage(p, type_)
    # uni_dis, count = len(np.unique(steps[:, 2]))
    # print(uni_dis)
    for i in range(len(steps)):
        u1, v1 = sorted([steps[i][0], steps[i][1]])
        u2, v2 = sorted([linked[i][0], linked[i][1]])
        d1, d2 = steps[i][2], linked[i][2]
        c1, c2 = steps[i][3], linked[i][3]
        if not (u1 == u2 and v1 == v2):
            indices = np.where(linked[:, 2] == d2)[0]
            # print(indices)
            if indices.shape[0] == 1:
                if abs(c1-c2)>1: # 针对观察得出的 SPJ
                    '''在观察里 print(indices): [574 575] [574 575] [1856]
                    同权值下，这意味着它们顺序合并有差异，导致新簇编号差异为1，这是由于未规定距离相同输出顺序的正常现象'''
                    raise Exception(f'failed1 {i} {u1} {v1} {u2} {v2}')
            else:
                ok = False
                for j in indices:
                    if sorted([linked[j][0], linked[j][1]]) == [u2, v2]:
                        ok = True
                        break
                if not ok:
                    raise Exception(f'failed2 {i} {u1} {v1} {u2} {v2}')
        assert np.abs(d1-d2)<1e-6
        assert c1 == c2
        # for j in range(4):
        #     assert linked[i][j] == steps[i][j], f'failed {i} {j} {linked[i][j]} {steps[i][j]}'

def checkAllStepsReconstruction():
    '''检查全部重构绘图的steps是否满足绘图库函数标准，直接读取结果'''
    p = utils.readCSV()
    for type_ in ALL_TYPES:
        with open(f'steps_{type_}.txt', 'r') as f:
            steps = eval(f.read())
        # print(type_)
        steps2 = BuildPlottingSteps(steps)
        try:
            checkStepsReconstruction(steps2, p, type_)
        except Exception as e:
            print(f'failed {type_} : {e}')
# checkAllStepsReconstruction()