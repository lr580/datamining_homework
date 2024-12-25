def checkSilhouette2():
    import numpy as np
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import silhouette_samples
    from sklearn.cluster import KMeans
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=1, random_state=42) # or n_c=3
    y_kmeans = kmeans.fit_predict(X) # [1 1 1 0 0 0]
    print(y_kmeans)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(silhouette_avg) # 整图
    silhouette_vals = silhouette_samples(X, y_kmeans)
    print(silhouette_vals) # 各点
# checkSilhouette2()

def testPlot2():
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例数据
    x = np.arange(0, 10, 0.1)
    y1 = np.sin(x)  # 第一个数据集
    y2 = np.exp(x / 5)  # 第二个数据集

    fig, ax1 = plt.subplots()

    # 绘制第一个折线图
    ax1.plot(x, y1, 'b-', label='sin(x)')
    ax1.set_xlabel('X轴')
    ax1.set_ylabel('sin(x)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    ax2.plot(x, y2, 'r-', label='exp(x/5)')
    ax2.set_ylabel('exp(x/5)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # 添加图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 显示图形
    plt.title('双y轴折线图示例')
    plt.show()
    
def testPlot3():
    import numpy as np
    import matplotlib.pyplot as plt

    # 生成一些示例数据
    n = 5000
    points = np.random.rand(n, 2)
    labels = np.random.randint(0, 15, n)  # 15个类别

    # 使用tab20 colormap
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab20', alpha=0.7)

    plt.title('Scatter Plot with Different Colors for Each Class')
    plt.colorbar(label='Class')
    plt.show()
# testPlot3()

def testPlot4():
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建一些数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # 创建子图
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # 使用 plt.title() 为每个子图添加独立标题
    axs[0].plot(x, y1)
    axs[0].set_title('正弦波')

    axs[1].plot(x, y2)
    axs[1].set_title('余弦波')

    # 使用 plt.suptitle() 添加整体标题
    plt.suptitle('三角函数图', fontsize=16)

    # 显示图形
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 确保整体标题不被遮挡
    plt.show()
testPlot4()