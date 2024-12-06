# 个人实现 https://github.com/lr580/algorithm_template
# 因为不给用第三方库，所以没用 sortedcontainer 的 sortedlist
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
        