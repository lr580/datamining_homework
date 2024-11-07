# 基数排序，一种 O(nc) 的排序算法，其中 c 较低，可以优于 O(nlogn) 
# 参考：https://oi-wiki.org/basic/radix-sort/
# 个人实现 https://github.com/lr580/algorithm_template
def radixSort(a, maxbit = 40, eachbit = 8):
    # 复杂度 O(nc), n=len(a), c=log2(max(a)) / eachbit
    mask=(1<<eachbit) - 1 
    x=a[::]
    n=len(a)
    b=[0]*n
    cnt = []
    for i in range(0, maxbit, eachbit):
        cnt = [0 for i in range(mask+1)] # 计数排序清零
        for x in a:
            cnt[(x>>i)&mask] += 1
        s = 0 # 计数排序变式
        for j in range(mask+1):
            s += cnt[j]
            cnt[j] = s - cnt[j]
        for x in a:
            idx=(x>>i)&mask
            b[cnt[idx]]=x
            cnt[idx]+=1
        a,b=b,a
        for i in range(n-1): # 剪枝
            if a[i]>a[i+1]: break
        else: continue
        break
    return a

# 效率对比 -> 证明了该排序比原生 O(nlogn) 排序更优
# from random import randint
# import time
# a = [randint(0, 2**40-1) for i in range(int(8e6))]
# b = a[::]
# import numpy as np
# c = np.array(a)
# t1 = time.perf_counter()
# radixSort(a)
# t2 = time.perf_counter()
# b.sort()
# t3 = time.perf_counter()
# c.sort()
# t4 = time.perf_counter()
# print(t2-t1, t3-t2, t4-t3) # 2.28s < 5.02s, 0.55s