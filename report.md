## Exercise \#1

> by 24214860 覃梓鑫

### 层次聚类

实现过程：

1. 使用 Python，基于手写加权并查集、最小堆等数据结构与算法，以 $O(n^2\log n)$ 时间、 $O(n^2)$ 空间复杂度优秀地实现了 min、max、group average、ward 四种层次聚类方法。(见 `cluster.py` 和辅助代码 `disjointSet.py`, `utils.py`)
2. 使用 `numpy` 数组，重构堆等方法思路，对代码进行了充分的效率优化，使代码时间空间开销明显降低。优化后，空间节省至少一半，运行时间从上百秒优化到分别 3.2(min)、10.8(max)、26.5(avg)、14.0(ward)秒。(运行CPU：i7-13620H)

运行结果在 `steps_min.txt` 等共四个 `steps_xxx.txt` 文件展示，参见 `cluster.py` 的 `cluster()` 注释例子解释含义。

### GMM

