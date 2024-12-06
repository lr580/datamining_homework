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

class disMatrix:
    '''给定欧氏距离a[n][2]点集，使用[][]运算符求任意下标(i,j)距离'''
    def __init__(self, a):
        ...