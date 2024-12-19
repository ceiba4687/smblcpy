import numpy as np


def linefit(n, vel):
    """
    线性拟合函数，计算数据序列的起点和终点值

    参数说明:
    n: int, 数据点数量
    vel: array, 输入数据序列

    返回值:
    al: float, 起点值（第一个点的拟合值）
    bl: float, 终点值（最后一个点的拟合值）
    """

    # 初始化矩阵和向量
    bat = np.zeros(2)
    mat = np.zeros((2, 2))

    # 计算矩阵元素
    bat[0] = np.sum(vel)
    mat[0, 0] = float(n)

    # 计算带权重的和
    x = np.arange(n)
    bat[1] = np.sum(vel * x)
    mat[0, 1] = np.sum(x)
    mat[1, 0] = mat[0, 1]
    mat[1, 1] = np.sum(x * x)

    # 计算行列式
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]

    # 检查奇异性
    if det == 0:
        raise ValueError("Error in linefit (singularity problem)!")

    # 计算拟合参数
    al = (mat[1, 1] * bat[0] - mat[0, 1] * bat[1]) / det
    bl = (mat[0, 0] * bat[1] - mat[1, 0] * bat[0]) / det

    # 计算终点值
    bl = al + bl * float(n - 1)

    return al, bl
