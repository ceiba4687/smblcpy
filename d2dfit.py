import numpy as np


def d2dfit(n, dis, p, b, ndeg, disfit):
    """
    多项式拟合函数

    参数说明:
    n: int, 数据点数量
    dis: array, 输入数据序列
    p: array, 勒让德多项式值矩阵 (n x (ndeg+1))
    b: array, 多项式系数
    ndeg: int, 多项式次数
    disfit: array, 拟合结果输出数组

    注意: 这个函数使用勒让德多项式(Legendre polynomials)进行拟合
    """

    # 计算x轴区间步长
    dx = 2.0 / float(n - 1)

    # 计算勒让德多项式值
    for i in range(n):
        x = -1.0 + float(i) * dx
        # 零阶项
        p[i, 0] = 1.0

        if ndeg > 0:
            # 一阶项
            p[i, 1] = x
            # 高阶项递推
            for ideg in range(2, ndeg + 1):
                p[i, ideg] = (
                    (2 * ideg - 1) * x * p[i, ideg - 1] - (ideg - 1) * p[i, ideg - 2]
                ) / float(ideg)

    # 计算多项式系数
    for ideg in range(ndeg + 1):
        # 使用梯形法则计算积分
        b[ideg] = 0.5 * (dis[0] * p[0, ideg] + dis[n - 1] * p[n - 1, ideg])
        for i in range(1, n - 1):
            b[ideg] += dis[i] * p[i, ideg]
        b[ideg] = b[ideg] * dx * 0.5 * float(2 * ideg + 1)

    # 计算拟合值
    for i in range(n):
        x = -1.0 + float(i) * dx
        disfit[i] = 0.0
        for ideg in range(ndeg + 1):
            disfit[i] += b[ideg] * p[i, ideg]

    return
