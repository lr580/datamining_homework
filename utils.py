def print2Darray(a, extraLine = True):
    '''调试用，输出二维浮点数组a[n][n]，带表头'''
    n = len(a)
    print('     ', end='')
    for i in range(n):
        print(str(i).center(4),end=' ')
    print()
    for i in range(n):
        print((str(i)+":").center(4), end=' ')
        for j in range(n):
            print(f'{a[i][j]:.2f}', end=' ')
        print()
    if extraLine:
        print()

def discretization(a):
    '''调试用，把离散取值变成连续取值，如a=[0,3,3,5,0]返回[0,1,1,2,0]，按出现顺序排值，因为不排序故复杂度O(n)'''
    res = dict()
    for x in a:
        if x not in res:
            res[x] = len(res)
    return [res[x] for x in a]

def reconstruct_points(distance_matrix):
    '''根据欧式距离矩阵，重构出可能的一种二维点方案 \n 
    主要用于验证课堂 PPT 例子，不在正式代码使用'''
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

class disMatrix:
    '''给定欧氏距离a[n][2]点集，使用[][]运算符求任意下标(i,j)距离'''
    def __init__(self, a):
        ...