# 手写实现的各种通用辅助函数
import time
from functools import wraps
import sys
import io
import numpy as np # 主要用于优化，不然太慢了
# import matplotlib as plt # 用于画图
# import subprocess

def print2Darray(a, extraLine = True):
    '''调试用，输出二维浮点数组a[n][m]，带表头'''
    n, m = len(a), len(a[0])
    print('     ', end='')
    for i in range(m):
        print(str(i).center(4),end=' ')
    print()
    for i in range(n):
        print((str(i)+":").center(4), end=' ')
        for j in range(m):
            print(f'{a[i][j]:.2f}', end=' ')
        print()
    if extraLine:
        print()

def readCSV(filepath = '8gau.txt'):
    '''读取作业要求的指定格式数据集，返回 [5000][2] 的点集列表 \n
    数据来源 https://www.cse.cuhk.edu.hk/~taoyf/course/cmsc5724/data/8gau.txt '''
    p = []
    with open(filepath) as f:
        for line in f.readlines():
            try:
                p.append([float(x) for x in line.split()])
            except:
                break
    return np.array(p)

def print_exec_time(func):
    '''装饰器，统计执行函数的用时'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs)  
        end_time = time.time() 
        execution_time = end_time - start_time 
        print(f"'{func.__name__}' {execution_time:.4f}s")
        return result
    return wrapper

def Timer():
    '''输出距离上一次调用该函数的时间间隔的函数闭包'''
    last_time = [time.time()]  # 使用列表存储上一次调用的时间
    def elapsed_time():
        current_time = time.time()
        interval = current_time - last_time[0]  # 计算时间间隔
        last_time[0] = current_time  # 更新上一次调用的时间
        print('%.4fs' % interval)
    return elapsed_time

def MemoryTracker():
    '''获取当前内存使用量(MB)'''
    import psutil
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    def track_memory():
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024) 
        memory_change = current_memory - initial_memory
        print(f"{memory_change:.3f}MB")
    return track_memory

# @print_exec_time
def getDisMatrix(p):
    '''根据点集p[n][2]生成距离矩阵，使用 numpy 优化'''
    # return [[(p[i][0] - p[j][0]) ** 2 + (p[i][1] - p[j][1]) ** 2 for j in range(len(p))] for i in range(len(p))] # 4.6598s
    a = np.array(p)
    return np.linalg.norm(a[:, np.newaxis] - a[np.newaxis, :], axis=2) # 0.6036

def getDisMatrixSquare(p):
    '''根据点集p[n][2]生成距离矩阵，使用 numpy 优化，返回距离的平方'''
    a = np.array(p)
    return np.sum((a[:, np.newaxis] - a[np.newaxis, :]) ** 2, axis=2)
# data = readCSV()
# getDisMatrix(data) 

def z_score(x):
    '''对输入点X[n][2],将其标准化为均值0方差1: 返回 (X-μ)/σ'''
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    return (x - mu) / sigma


# 下面代码在正式里没有使用
def getGMMsampleParams():
    '''随机获得一个固定的高斯混合模型参数，用于测试算法正确性'''
    z = np.array([0.1, 0.6, 0.3])
    m = np.array([[0, 0], [0, 7], [7, 0]])
    c = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]])
    # 测试数据范围对结果的影响：结果表明，在下面情况下，我手写的GMM会NAN
    # m = np.array([[0, 0], [0, 7], [7, 0]]) * 1e5
    # c = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]]) * 1e10
    return z, m, c, 5000

def getGMMsampleParams2():
    '''随机获得一个固定的高斯混合模型参数，用于测试算法正确性'''
    # 测试数据范围对结果的影响：结果表明，在下面情况下，我手写的GMM会NAN
    z = np.array([0.1, 0.6, 0.3])
    m = np.array([[0, 0], [0, 7], [7, 0]]) * 1e5
    c = np.array([[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]], [[1, 0], [0, 1]]]) * 1e10
    return z, m, c, 5000

def getGMMsamplePoints(z, m, c, n, seed=580, shuffle=True):
    '''给定占比z[k],均值m[k][2],协方差矩阵c[k][2][2]和点数n \n
    生成高斯混合模型样本点集X[n][2],标签y[n] \n
    如果shuffle=True，随机打乱数据返回'''
    np.random.seed(seed)
    k = len(z)
    n_samples = (n * z).astype(int)
    n_samples[-1] += n - sum(n_samples)
    X, y = [], []
    for label in range(k):
        X.append(np.random.multivariate_normal(m[label], c[label], n_samples[label]))
        y.extend([label] * n_samples[label])
    X = np.vstack(X)
    y = np.array(y)
    if shuffle:
        indices = np.arange(n)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    return X, y

def chcp():
    '''切换中文编码，已废置'''
    # subprocess.run('chcp 65001', shell=True)
    # 设置标准输出为 UTF-8 编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# print2Darray(readCSV())
# assert len(readCSV()) == 5000 and len(readCSV()[0]) == 2

def reconstruct_points(distance_matrix):
    '''根据欧式距离矩阵，重构出可能的一种二维点方案 \n 
    主要用于验证课堂 PPT 例子，不在正式代码使用 \n 
    原理参考高等工程数学，对角分解和特征值'''
    import numpy as np
    if isinstance(distance_matrix, list):
        distance_matrix = np.array(distance_matrix)
    n = distance_matrix.shape[0]
    D_squared = distance_matrix ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_squared @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    k = 2
    X = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    return X

class DisMatrixRow:
    '''中间类，给下文 disMatrix 实现 [][] 使用'''
    def __init__(self, disMatrix, i):
        self.dm = disMatrix
        self.i = i
    def __getitem__(self, j):
        return self.dm._distance(self.i, j)

class DisMatrix:
    '''已废置，在优化过的代码中不需要使用该类 \n
    给定欧氏距离a[n][2]点集，使用[][]运算符求任意下标(i,j)距离 \n
    一种模拟出n阶距离矩阵，但实际空间为 O(n) 的压缩矩阵存储 \n
    主要是为了兼容课堂 PPT 的例子测试，使输入可以是点集也可以是距离矩阵 \n
    使用示例：\n
    dm = DisMatrix([[0,0], [3,4]]) \n
    print(dm[0][1]) #期望输出 5，因为 (0,0), (3,4) 距离为 5'''
    def __init__(self, p):
        '''输入点集p[n][2]以初始化'''
        self.p = p
    def _distance(self, i, j):
        dx = self.p[i][0] - self.p[j][0]
        dy = self.p[i][1] - self.p[j][1]
        return (dx ** 2 + dy ** 2) ** 0.5
    def __getitem__(self, index):
        return DisMatrixRow(self, index)
    def __len__(self):
        return len(self.p)

# dm = DisMatrix([[0,0], [3,4]])
# print(dm[0][1])

def getPPTsampleMatrix():
    '''返回PPT课件58页的距离矩阵样例'''
    return [
        [0, 0.24, 0.22, 0.37, 0.34, 0.23],
        [0.24, 0, 0.15, 0.20, 0.14, 0.25],
        [0.22, 0.15, 0, 0.15, 0.28, 0.11],
        [0.37, 0.20, 0.15, 0, 0.29, 0.22],
        [0.34, 0.14, 0.28, 0.29, 0, 0.39],
        [0.23, 0.25, 0.11, 0.22, 0.39, 0]]

# 测试例子
# pm = getPPTsampleMatrix()
# p = reconstruct_points(pm)
# print(getDisMatrix(p))

def discretization(a):
    '''调试用，把离散取值变成连续取值，如a=[0,3,3,5,0]返回[0,1,1,2,0]，按出现顺序排值，因为不排序故复杂度O(n)'''
    res = dict()
    for x in a:
        if x not in res:
            res[x] = len(res)
    return [res[x] for x in a]

def radixSort(a, maxbit = 40, eachbit = 8):
    '''基数排序，一种 O(nc) 的排序算法，其中 c 较低，可以优于 O(nlogn) ；但是表现不如 np.sort 故废置 \n
    参考：https://oi-wiki.org/basic/radix-sort/ \n
    个人实现 https://github.com/lr580/algorithm_template \n
    复杂度 O(nc), n=len(a), c=log2(max(a)) / eachbit'''
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