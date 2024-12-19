import numpy as np


def rampfit(n, y, n1, n2, i1, i2, y0, smin, swap):
    """
    斜坡函数拟合

    参数说明:
    n: int, 数据长度
    y: array, 输入数据序列
    n1: int, 起始点索引
    n2: int, 结束点索引
    i1: int, 最优斜坡起始点(输出)
    i2: int, 最优斜坡结束点(输出)
    y0: float, 最优斜率(输出)
    smin: float, 最小误差(输出)
    swap: array, 工作数组 (n x 2)
    """

    # 计算向后累加和
    swap[n - 1, 0] = y[n - 1]
    for i in range(n - 2, n1 - 1, -1):
        swap[i, 0] = swap[i + 1, 0] + y[i]

    # 计算加权向前累加和
    swap[0, 1] = y[0]
    for i in range(1, n2):
        swap[i, 1] = swap[i - 1, 1] + (i + 1) * y[i]

    # 初始化最优解
    i2 = n1
    i1 = i2 - 1
    y0 = swap[i2, 0] / float(1 + n - i2)
    smin = -(y0**2) * float(1 + n - i2)

    # 设置搜索范围
    j1min = n1
    j1max = n2
    j2min = n1
    j2max = n2
    id = 1 + (n2 - n1) // 50

    # 迭代搜索最优解
    while True:
        for j2 in range(j2min + 1, j2max + 1, id):
            y0_temp = swap[j2, 0] / float(1 + n - j2)
            delta = -(y0_temp**2) * float(1 + n - j2)

            for j1 in range(j1min, min(j1max + 1, j2), id):
                # 计算误差
                sigma = delta + (
                    y0_temp * float(j2 - j1 - 1) * float(2 * (j2 - j1) - 1) / 6.0
                    + 2.0
                    * (
                        float(j1 + 1) * (swap[j1, 0] - swap[j2 - 1, 0])
                        - swap[j2 - 1, 1]
                        + swap[j1, 1]
                    )
                ) * y0_temp / float(j2 - j1)

                # 更新最优解
                if smin > sigma:
                    i1 = j1
                    i2 = j2
                    smin = sigma

        # 如果搜索间隔大于1，缩小搜索范围继续迭代
        if id > 1:
            j1min = max(n1, i1 - 5 * id // 2)
            j1max = min(n2, i1 + 5 * id // 2)
            j2min = max(n1, i2 - 5 * id // 2)
            j2max = min(n2, i2 + 5 * id // 2)
            id = 1 + id // 5
        else:
            break

    # 计算总误差
    smin = smin + np.sum(y**2)

    # 返回最优解
    y0 = swap[i2, 0] / float(1 + n - i2)
    return i1, i2, y0, smin
