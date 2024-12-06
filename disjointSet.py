# 并查集实现，均摊 O(α)≈O(1) 复杂度的单次操作；参考：https://oi-wiki.org/ds/dsu-complexity
# 个人实现 https://github.com/lr580/algorithm_template

class DSU:
    '''朴素并查集'''
    def __init__(self, n):
        self.fa = [i for i in range(n)] # 根节点
    def findFa(self, x):
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]
            x = self.fa[x] # 路径压缩
        return x
    def merge(self, x, y):
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        self.fa[fx] = fy
        return True
        
# 代码正确性检验 https://leetcode.cn/problems/number-of-islands
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

class DSU_max:
    '''最大聚类并查集'''
    def __init__(self, n, a):
        self.n = n
        self.fa = [i for i in range(n)] # 根节点
        self.max = a # 与每组距离的最大值
    def findFa(self, x):
        if x == self.fa[x]:
            return x
        res = self.findFa(self.fa[x])
        for i in range(self.n):
            self.max[x][i] = max(self.max[x][i], self.max[self.fa[x]][i])
        print(self.max[x], self.max[self.fa[x]], x, self.fa[x])
        self.fa[x] = res
        return self.fa[x]
    def merge(self, x, y):
        fx, fy = self.findFa(x), self.findFa(y)
        if fx == fy:
            return False
        self.fa[fx] = fy
        return True

class DSU_size:
    '''维护节点元素数、已用边的并查集 \n'''
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