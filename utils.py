# 手写实现的各种通用辅助函数

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
    '''读取作业要求的指定格式数据集，返回 [5000][2] 的点集列表'''
    p = []
    with open(filepath) as f:
        for line in f.readlines():
            try:
                p.append([float(x) for x in line.split()])
            except:
                break
    return p

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

class disMatrixRow:
    '''中间类，给下文 disMatrix 实现 [][] 使用'''
    def __init__(self, disMatrix, i):
        self.dm = disMatrix
        self.i = i
    def __getitem__(self, j):
        return self.dm._distance(self.i, j)

class disMatrix:
    '''给定欧氏距离a[n][2]点集，使用[][]运算符求任意下标(i,j)距离 \n
    一种模拟出n阶距离矩阵，但实际空间为 O(n) 的压缩矩阵存储 \n
    主要是为了兼容课堂 PPT 的例子测试，使输入可以是点集也可以是距离矩阵 \n
    使用示例：\n
    dm = disMatrix([[0,0], [3,4]]) \n
    print(dm[0][1]) #期望输出 5，因为 (0,0), (3,4) 距离为 5'''
    def __init__(self, p):
        '''输入点集p[n][2]以初始化'''
        self.p = p
    def _distance(self, i, j):
        dx = self.p[i][0] - self.p[j][0]
        dy = self.p[i][1] - self.p[j][1]
        return (dx ** 2 + dy ** 2) ** 0.5
    def __getitem__(self, index):
        return disMatrixRow(self, index)

# dm = disMatrix([[0,0], [3,4]])
# print(dm[0][1])

def discretization(a):
    '''调试用，把离散取值变成连续取值，如a=[0,3,3,5,0]返回[0,1,1,2,0]，按出现顺序排值，因为不排序故复杂度O(n)'''
    res = dict()
    for x in a:
        if x not in res:
            res[x] = len(res)
    return [res[x] for x in a]