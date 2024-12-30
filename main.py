import sys
import cluster_criteria
import cluster
# 项目内容详见文档 README.md 和代码注释
is_unk = False
def unknown():
    print('未知参数，请参考 README.md 传入参数选择要执行的功能，或检查你的输入是否有误。')
    global is_unk
    is_unk = True
def main():
    argv = sys.argv[1:]
    show = not (len(argv) >= 3 and argv[2] == '--save') # 是展示开始写文件
    if not argv or argv[0] == '--check':
        print('OK, 项目正常运行！\n请参考 README.md 传入参数选择要执行的功能。')
        return
    print('计算中，运行可能需要一段时间，请稍等。')
    if argv[0] == '--cluster': # 层次聚类
        cluster.generateAllClusterSteps(True) # 如果没有计算过，先计算一下
        if argv[1] == 'plot': # 绘制4种聚类的散点图
            cluster_criteria.plotAllTypesCluster('cluster_results.png', show=show)
        elif argv[1] == 'stat': # 计算SSE, 轮廓系数，绘图展示
            print('由于轮廓系数计算较慢，请耐心等待大约一两分钟。')
            cluster_criteria.plotLines('both', show=show)
        elif argv[1] == 'step': # 展示层次聚类步骤，保存到磁盘
            cluster_criteria.plotAllSteps()
            show = False
        elif argv[1] == 'generate': # 进行层次聚类，得到聚类步骤结果，用于后续绘图
            cluster.generateAllClusterSteps() # 就算计算过，也重新计算一下
            show = False
        else:
            unknown()
    elif argv[0] == '--gmm': # GMM
        if argv[1] == 'diff_init': # 绘图展示比较不同初始化方式
            cluster_criteria.plotLines_GMM(show=show)
        elif argv[1] == 'diff_k': # 绘图展示比较不同聚类数
            cluster_criteria.plotLines_GMM_k(show=show)
        elif argv[1] == 'diff_k_stat': # 比较不同聚类数的SSE，轮廓系数
            cluster_criteria.plotSSE_Silhouette_GMM_k(show=show)
        else:
            unknown()
    elif argv[0] == '--compare': # 层次聚类与GMM比较
        if argv[1] == 'plot': # 对比绘图展示层次聚类和GMM聚类结果
            cluster_criteria.plotWardVsGMM(show=show)
        elif argv[1] == 'stat': # 对比展示层次聚类和GMM聚类的SSE，轮廓系数
            cluster_criteria.compareWardAndGMM(show=show)
    else:
        unknown()
    if not show and not is_unk: # 不是在线展示而是写入磁盘
        print('结果已保存到磁盘，请到项目根目录查看！')
main()