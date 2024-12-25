# 并查集（DSU，Disjoint Set Union）个人手写实现
# 均摊 O(α)≈O(1) 复杂度的单次操作；复杂度分析参考：https://oi-wiki.org/ds/dsu-complexity
# 基于我个人的算法模板集 https://github.com/lr580/algorithm_template
import copy
import numpy as np

class DSU:
    '''朴素并查集，可证明平均单次操作复杂度为 O(α)≈O(1)，α 是反阿克曼函数'''
    def __init__(self, n):
        self.fa = [i for i in range(n)] # 根节点
        self.n = n
    def findFa(self, x):
        '''求x节点的根并返回；比递归和先求再压缩都快'''
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x] # 路径压缩
        return x
    def mergeop(self, fx, fy):
        '''钩子函数，额外信息合并，给定两个根节点fx->fy'''
    def merge(self, x, y):
        '''若两节点x,y不在同一根，合并并返回True，否则返回False'''
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        fy, fx = sorted([fx, fy]) # 最小做根，避免 DSU_virtual 子类的越界
        self.mergeop(fx, fy) # 钩子函数，给子类用
        self.fa[fx] = fy
        return True
    # def fas(self, x, y):
    #     return self.findFa(x), self.findFa(y)
        
# 代码正确性检验如下：检测链接 https://leetcode.cn/problems/number-of-islands
'''
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n = 0
        m = dict()
        for i, r in enumerate(grid):
            for j, v in enumerate(r):
                if v == '1':
                    m[(i,j)] = n
                    n += 1
        ds = DSU(n)
        r, c = len(grid), len(grid[0])
        for x,y in m:
            for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
                ax, ay = x+dx, y+dy
                if 0<=ax<r and 0<=ay<c and (ax,ay) in m:
                    ds.merge(m[(x,y)], m[(ax,ay)])
        return sum([ds.findFa(i)==i for i in range(n)])
'''

def getClasses(dsu:DSU):
    '''把DSU的[0,n)分类映射到[0,k)返回长n列表代表每个元素属于哪类'''
    n = len(dsu.fa)
    maps = dict()
    for i in range(n):
        if dsu.findFa(i) == i:
            maps[dsu.findFa(i)] = len(maps)
    return np.array([maps[dsu.fa[i]] for i in range(n)])
   
class DSU_virtual(DSU):
    '''每次合并将一个虚拟节点加入合并后的类(假设最多合并m次)，并维护大小'''
    def __init__(self, n, m):
        self.n = n
        self.top = n-1 
        ''' self.top: 当前最大点，作用是与cluster代码的堆结合，用top来判断当前类，从而起到vis数组的作用来排除堆里在同一个类的点的判断；这是因为：一旦 u,v 合并，那么在堆里，用原本 u,v 的数据都是旧数据，合并后的新距离数据才是有效的，通过 top 来反映当前堆的元素是新的还是过时的，实验表明，这样的设计大大优化时空效率 (参见 cluster.py 的 averageCluster 和 wardCluster)'''
        self.fa = np.arange(n+m) # self.fa = [i for i in range(n+m)] # 根节点
        self.siz = np.ones(n) # self.siz = [1 for i in range(n)] # 当前类元素数
    def mergeop(self, fx, fy):
        self.top += 1
        self.fa[self.top] = fy
        self.siz[fy] += self.siz[fx]
        # self.siz[fx] = 0
        
class DSU_average(DSU_virtual):
    '''维护组间信息的并查集，维护组元素数目，\n并且每次合并将一个虚拟节点加入合并后的类(假设最多合并m次) \n 专门适配层次聚类的并查集'''
    def __init__(self, n, m, a):
        super().__init__(n, m)
        self.e = a
    def mergeop(self, fx, fy):
        super().mergeop(fx, fy)
        self.e[fy, :] += self.e[fx, :]
        self.e[:, fy] += self.e[:, fx]
        # self.e[fx, :] = self.e[:, fx] = 0
        
class DSU_ward(DSU_virtual):
    ''' DSU_avg + DSU_virtual，参见二者文档'''
    def __init__(self, n, m, p):
        super().__init__(n, m)
        self.sx = np.copy(p[:, 0])  # 当前簇 x 坐标和
        self.sy = np.copy(p[:, 1])  # 当前簇 y 坐标和
    def mergeop(self, fx, fy):
        super().mergeop(fx, fy)
        self.sx[fy] += self.sx[fx]
        self.sy[fy] += self.sy[fx]
        # self.sx[fx] = self.sy[fx] = 0
    def avg(self, fx): # 求簇的平均值点
        n = self.siz[fx]
        return self.sx[fx] / n, self.sy[fx] / n

class DSU_hard(DSU):
    '''每次合并后强制更新全体fa数组'''
    def __init__(self, n):
        super().__init__(n)
        self.ele = [set([i]) for i in range(n)] # 元素集
        self.num = n # 簇个数
    def mergeop(self, fx, fy):
        self.ele[fy].update(self.ele[fx])
        for x in self.ele[fx]:
            self.fa[x] = fy # self.findFa(x)
        self.ele[fx].clear() # 节省内存
        self.num -= 1
    
class DSU_SSE(DSU):
    '''复杂度良好实时计算SSE的并查集 \n
    公式(参见高等工程数学)：(x-μ)^2=x^2-2xμ+Nμ^2=x^2-Nμ^2 \n
    也可以用 DSU_hard 做子类，就不额外维护一个 siz 参见其他上文代码'''
    def __init__(self, n, p):
        super().__init__(n)
        self.sx = np.copy(p[:, 0])  # 当前簇 x 坐标和
        self.sy = np.copy(p[:, 1])  # 当前簇 y 坐标和
        self.sx2 = p[:, 0] ** 2 # 当前簇 x 坐标平方和
        self.sy2 = p[:, 1] ** 2 # 当前簇 y 坐标平方和
        self.siz = np.ones(n)
        self.se = np.zeros(n)
        self.sse = 0.
    def avg(self, x): #x=fx，求中心
        n = self.siz[x]
        return self.sx[x] / n, self.sy[x] / n
    def variance(self, u): # 求并保存当前类x=fx的方差 (x-μ)^2
        ax, ay = self.avg(u)
        vx = self.sx2[u] - self.siz[u] * ax ** 2
        vy = self.sy2[u] - self.siz[u] * ay ** 2
        self.se[u] = vx+vy
        return self.se[u]
    def mergeop(self, fx, fy):
        super().mergeop(fx, fy)
        self.sse -= self.se[fx]
        self.sse -= self.se[fy]
        self.sx[fy] += self.sx[fx]
        self.sy[fy] += self.sy[fx]
        self.sx2[fy] += self.sx2[fx]
        self.sy2[fy] += self.sy2[fx]
        self.siz[fy] += self.siz[fx]
        self.sse += self.variance(fy)
''' 测试用例：
dsu = DSU_SSE(4, np.array([[1,2],[2,3],[5,6],[6,7]]))
print(dsu.sse) # 0
dsu.merge(0,1)
print(dsu.sse) # 1
dsu.merge(2,3)
print(dsu.sse) # 2
dsu.merge(0,2)
print(dsu.sse) # 34
# 与答案对比
import numpy as np
from sklearn.cluster import KMeans
data = np.array([[1, 2], [2, 3], [5, 6], [6, 7]])
kmeans = KMeans(n_clusters=1, random_state=0).fit(data)
print(kmeans.inertia_) # 34 (其他同理，1是1类，n_clusters=2可以输出2)
'''

# 往下的数据结构没有正式代码中用到，它们是在我实现层次聚类过程中的一些尝试
class DSU_cluster(DSU):
    '''维护组间信息的并查集，维护组元素数目，\n并且每次合并将一个虚拟节点加入合并后的类(假设最多合并m次) \n 专门适配层次聚类的并查集 \n 已经被拆分优化，以便于代码复用'''
    def __init__(self, n, m, a, f):
        '''f(a,fu,fv)是函数，当fu加入fv时，使用f函数聚合a数据'''
        self.n = n
        self.fa = np.arange(n+m) # self.fa = [i for i in range(n+m)] # 根节点
        self.siz = np.ones(n) # self.siz = [1 for i in range(n)] # 当前类元素数
        self.top = n-1 # 当前最大点
        self.e = a
        self.f = f
    def mergeop(self, fx, fy):
        self.f(self.e, fx, fy)
        self.top += 1
        self.fa[self.top] = fy
        self.siz[fy] += self.siz[fx]
        # self.siz[fx] = 0
    @staticmethod
    def add(a, fx, fy):
        '''求和聚合距离矩阵'''
        a[fy, :] += a[fx, :]
        a[:, fy] += a[:, fx]
        # a[fx, :] = a[:, fx] = 0

class DSU_ele(DSU):
    '''维护节点元素并查集，组间avg使用 \n 目前已废置，有更优化的实现'''
    def __init__(self, n):
        super().__init__(n)
        self.ele = [set([i]) for i in range(n)] # 元素集
        # self.info = copy.deepcopy(a) # 聚合信息集
        # self.agg = f
    def mergeop(self, fx, fy):
        # for i in range(self.n):
        #     self.info[fy][i] = self.agg(self.info[fx][i], self.info[fy][i])
        self.ele[fy] |= self.ele[fx]
        self.ele[fx] = set()

class DSU_avg(DSU_ele):
    '''维护各点和平均值的并查集，ward使用 \n 目前已废置，有更优化的实现'''
    def __init__(self, n, p):
        super().__init__(n)
        self.sx = [p[i][0] for i in range(n)] # 当前簇x坐标和
        self.sy = [p[i][1] for i in range(n)] # 当前簇y坐标和
    def mergeop(self, fx, fy):
        super().mergeop(fx, fy)
        self.sx[fy] += self.sx[fx]
        self.sy[fy] += self.sy[fx]
        self.sx[fx] = self.sy[fx] = 0
    def avg(self, fx): # 求簇的平均值点
        n = len(self.ele[fx])
        return self.sx[fx] / n, self.sy[fx] / n

class DSU_max(DSU):
    '''最大聚类并查集'''
    def __init__(self, n, a):
        super().__init__(n)
        self.max = copy.deepcopy(a) # 与每组距离的最大值
    def mergeop(self, fx, fy):
        for i in range(len(self.fa)):
            self.max[fy][i] = max(self.max[fx][i], self.max[fy][i])
    '''def merge(self, x, y):
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        # fx, fy = sorted([fx, fy], reverse=True) 加上这行发现样例不对，进而发现了bugs
        for i in range(len(self.fa)):
            self.max[fy][i] = max(self.max[fx][i], self.max[fy][i])
        self.fa[fx] = fy
        return True'''
    '''def findFa(self, x): # 能对，但是该写法太麻烦
        if x == self.fa[x]:
            return x
        res = self.findFa(self.fa[x])
        print(self.max[x], self.max[self.fa[x]], x+1, self.fa[x]+1)
        for i in range(self.n):
            self.max[x][i] = max(self.max[x][i], self.max[self.fa[x]][i])
        print(self.max[x])
        self.fa[x] = res
        return self.fa[x]'''

class DSU_size:
    '''维护节点元素数、已用边的并查集;暂时没用到这个并查集 \n 
    写这个数据结构时还没想到父子类代码复用'''
    def __init__(self, n):
        self.n = n
        self.fa = [i for i in range(n)] # 根节点
        self.size = [1 for i in range(n)] # 子树节点数
    def findFa(self, x):
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x] # 路径压缩
        return x
    def merge(self, x, y):
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        self.size[fx] += self.size[fy]
        self.size[fy] = 0
        self.used[fx] += self.used[fy]
        self.used[fy] = 0
        self.fa[fx] = fy
        return True
    
class DSU_rank(DSU):
    '''朴素并查集，按秩合并，没有采用'''
    def __init__(self, n):
        super().__init__(n)
        self.rank = [1] * n # 元素数
    def mergeop(self, fx, fy):
        '''钩子函数，额外信息合并，给定两个根节点fx->fy'''
        self.rank[fy] += 1
    def merge(self, x, y):
        '''若两节点x,y不在同一根，合并并返回True，否则返回False'''
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        # 总是把 fx 合并到 fy 去
        if self.rank[fx] > self.rank[fy]: 
            fx, fy = fy, fx
        self.mergeop(fx, fy) # 钩子函数，给子类用
        self.fa[fx] = fy
        return True
