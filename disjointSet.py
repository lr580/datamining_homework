# 并查集（DSU，Disjoint Set Union）个人手写实现
# 均摊 O(α)≈O(1) 复杂度的单次操作；复杂度分析参考：https://oi-wiki.org/ds/dsu-complexity
# 基于我个人的算法模板集 https://github.com/lr580/algorithm_template
import copy

class DSU:
    '''朴素并查集'''
    def __init__(self, n):
        self.fa = [i for i in range(n)] # 根节点
        self.n = n
    def findFa(self, x):
        '''求x节点的根并返回'''
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
        fx, fy = sorted([fx, fy], reverse=True) # 最小做根，方便debug输出信息
        self.mergeop(fx, fy) # 钩子函数，给子类用
        self.fa[fx] = fy
        return True
        
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
    return [maps[dsu.fa[i]] for i in range(n)]

class DSU_ele(DSU):
    '''维护节点元素并查集，组间avg使用'''
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
    '''维护各点和平均值的并查集，ward使用'''
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


# 往下的数据结构没有正式代码中用到，它们是在我实现层次聚类过程中的一些尝试

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
