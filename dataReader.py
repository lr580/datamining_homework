# 数据来源 https://www.cse.cuhk.edu.hk/~taoyf/course/cmsc5724/data/8gau.txt
def read8gau(path='8gau.txt'):
    '''返回 [5000, 2] 数组代表 5000 个空间点坐标'''
    data = []
    with open(path) as f:
        for line in f.readlines():
            nums = [int(i) for i in line.split()]
            if nums:
                data.append(nums)
    return data
# 数据观察 
# data = read8gau()
# print(max((max(i) for i in data))) # 970756
# print(min((min(i) for i in data))) # 19835
