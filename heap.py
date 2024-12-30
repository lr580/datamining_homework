# 个人手写实现的可删堆
# 目前优化过的层次聚类没使用 heap.py，该代码文件是我在实现层次聚类过程的尝试
# 基于我个人的算法模板集 https://github.com/lr580/algorithm_template
# 在优化过的层次聚类中，不再需要使用可删堆等数据结构，故目前废置
import numpy as np
from heapq import heappush, heappop
class Heap:
    '''以O(logn)均摊实现可以删除元素的最小堆 (小根堆)'''
    def __init__(self):
        self.a = [] # 懒删除的小根堆
        self.b = [] # 未处理的待删除元素
    def size(self):
        return len(self.a) - len(self.b)
    def insert(self, x):
        heappush(self.a, x)
    def erase(self, x):
        heappush(self.b, x)
    def top(self):
        while self.b and self.a[0] == self.b[0]:
            heappop(self.a)
            heappop(self.b)
        return self.a[0]
    def pop(self):
        x = self.top()
        heappop(self.a)
        return x

# 代码正确性检验如下：检测链接 https://leetcode.cn/problems/kth-largest-element-in-an-array
'''
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        h = Heap()
        for x in nums:
            h.insert(-x)
        #for i in range(k-1):
        #    h.pop()
        for x in sorted(nums, reverse=True)[:k-1]:
            h.erase(-x)
        return -h.top()
'''

class HeapMap:
    '''在可删堆Heap基础上，假设键值对是 (k,v)，按值v排序，添加功能：\n
    根据 k 寻找 v；实现对其的维护'''
    def __init__(self):
        self.heap = Heap()
        self.k2v = {}
    def add(self, k, v):
        assert k not in self.k2v
        self.k2v[k] = v
        self.heap.insert((v, k))
        print(len(self.k2v), len(self.heap.a), len(self.heap.b))
    def erase(self, k):
        if k in self.k2v:
            v = self.k2v[k]
            self.heap.erase((v, k))
            del self.k2v[k]
    def modify(self, k, v):
        self.erase(k)
        self.add(k, v)
    def getMin(self):
        '''取最小值的键值对k,v'''
        return tuple(reversed(self.heap.top())) 


class TreeMap:
    '''红黑树基础上，假设键值对是 (k,v)，按值v排序，添加功能：\n
    根据 k 寻找 v；实现对其的维护，键是矩阵(n,n) \n
    值是 (n, s) 表示总点数和总边权 \n
    专为avg层次聚类设计'''
    def __init__(self, n):
        from sortedcontainers import SortedList    
        self.n = n
        self.map = SortedList()
        self.k2n = np.zeros((n,n), dtype=np.int16)
        self.k2s = np.zeros((n,n), dtype=np.float64)
    def getDis(self,u,v):
        '''给定u<v，得到u,v两组的avg层次聚类平均边权'''
        return self.k2s[u,v] / self.k2n[u,v]
    def add(self, u, v, n, s):
        self.k2n[u,v] = n
        self.k2s[u,v] = s
        self.map.add((self.getDis(u, v), u, v))
    def erase(self, u, v):
        if self.k2n[u,v] > 0:
            self.map.remove((self.getDis(u, v), u, v))
            self.k2n[u,v] = 0
            self.k2s[u,v] = 0
    def modify(self, u, v, n, s):
        self.erase(u, v)
        self.add(u, v, n, s)
    def getMin(self):
        dis, u, v = self.map[0]
        return u, v, dis
    def get(self, u, v):
        '''返回 (n, s)'''
        return self.k2n[u,v], self.k2s[u,v]
    
def TreeMapEfficiencyTest():
    '''测试效率'''
    import utils
    import random
    timer = utils.Timer()
    n = 5000
    # a = SortedList()
    b = []
    for i in range(n):
        for j in range(i+1, n):
            # a.add(random.random()) # 1.17s(n=1000)
            # a.add((random.random(), i, j)) # 1.81s(n=1000) # 69.79s
            # heappush(b, random.random()) # 0.08s(n=1000)
            heappush(b, (random.random(), i, j)) #0.15s(n=1000) # 3.55s
    timer()
# TreeMapEfficiencyTest()