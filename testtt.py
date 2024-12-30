# import matplotlib.pyplot as plt
# import numpy as np

# labels = ['指标 1', '指标 2']
# result_A = [50, 80]
# result_B = [70, 60]
# x = np.arange(len(labels)) 
# width = 0.35 

# fig, ax = plt.subplots()
# bars1 = ax.bar(x - width/2, result_A, width, label='结果 A')
# bars2 = ax.bar(x + width/2, result_B, width, label='结果 B')

# ax.set_ylabel('值')
# ax.set_title('结果对比')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # 数据
# labels = ['指标 1', '指标 2']
# result_A = [100000, 0.8]
# result_B = [70000, 0.6]

# x = np.arange(len(labels))  # 指标的标签位置
# width = 0.35  # 柱子的宽度

# fig, ax1 = plt.subplots()

# # 绘制第一个指标的柱状图
# bars1 = ax1.bar(x - width/2, result_A, width, label='结果 A', color='b')
# bars2 = ax1.bar(x + width/2, result_B, width, label='结果 B', color='g')

# # 设置第一个坐标轴
# ax1.set_ylabel('指标 1 (1e5)', color='b')
# ax1.set_xlabel('指标')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels)
# ax1.tick_params(axis='y', labelcolor='b')

# # 创建第二个坐标轴
# ax2 = ax1.twinx()
# ax2.set_ylabel('指标 2 (0-1)', color='g')
# ax2.tick_params(axis='y', labelcolor='g')

# # 显示图例
# fig.tight_layout()
# ax1.legend(loc='upper left')
# ax2.legend(['结果 A', '结果 B'], loc='upper right')

# # 显示图表
# plt.title('结果对比（双坐标轴）')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# 数据
results = ['结果1', '结果2']
indicators1 = [80000, 60000]  # 第一个指标的值
indicators2 = [0.8, 0.6]       # 第二个指标的值

# 柱状图的宽度
bar_width = 0.35
x = np.arange(len(results))

# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 绘制第一个指标的柱状图
ax1.bar(x - bar_width/2, indicators1, width=bar_width, label='指标1', color='b')
ax1.set_ylabel('指标1 (范围: 0 - 100000)')
ax1.set_xticks(x)
ax1.set_xticklabels(results)

# 创建第二个坐标轴
ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, indicators2, width=bar_width, label='指标2', color='r')
ax2.set_ylabel('指标2 (范围: 0 - 1)')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 显示图形
plt.title('结果对比柱状图')
plt.show()