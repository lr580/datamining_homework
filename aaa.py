import utils
import numpy as np
# from numba import njit
import random
timer = utils.Timer()
memoryer = utils.MemoryTracker()
# a = np.random.rand(5000,5000) # -> 12s 194.26MB
a = [[random.random() for i in range(5000)] for j in range(5000)] # -> 14s 966.79MB
s=0
for i in range(5000*4999//2):
    u=random.randint(0,4999)
    v=random.randint(0,4999)
    s+=a[u][v]
timer()
memoryer()