import numpy as np
from d2dfit import d2dfit
from rampfit import rampfit
from typing import Tuple
import numpy as np
import numpy.typing as npt


def bscmono(
    nwin: int,
    ipre: int,
    ipga: int,
    isdw: int,
    vel: npt.NDArray[np.float64],
    err: npt.NDArray[np.float64],
    dt: float,
):
    """
    单调基线校正函数

    参数说明:
    nwin: int, 时间窗口长度
    ipre: int, 预事件窗口终点
    ipga: int, 最大加速度时刻
    isdw: int, 信号窗口终点
    vel: array, 速度记录
    err: array, 误差记录
    dt: float, 采样时间间隔

    返回值:
    offset: float, 基线偏移值
    rbserr: float, 基线校正误差
    """
    # 初始化参数
    ndat = isdw - ipre + 1
    sigma = np.zeros(ndat, dtype=np.float64)
    beta = np.zeros(ndat, dtype=np.float64)

    # 计算信号窗口内的速度变化率
    for i in range(ndat):
        sigma[i] = vel[i + ipre] - vel[i + ipre - 1]
        beta[i] = sigma[i] / dt

    # 寻找速度最大变化率
    imax = np.argmax(np.abs(beta))
    bmax = beta[imax]

    # 计算基线校正参数
    if abs(bmax) > 0:
        # 标准化变化率
        beta /= bmax

        # 计算累积效应
        cumbeta = np.zeros(ndat, dtype=np.float64)
        cumbeta[0] = beta[0]
        for i in range(1, ndat):
            cumbeta[i] = cumbeta[i - 1] + beta[i]

        # 计算单调性指标
        mono = np.zeros(ndat, dtype=np.float64)
        for i in range(ndat):
            if i < imax:
                mono[i] = min(1.0, max(0.0, cumbeta[i]))
            else:
                mono[i] = min(1.0, max(0.0, 2.0 - cumbeta[i]))

        # 计算基线偏移和校正误差
        offset = 0.0
        rbserr = 0.0
        count = 0
        for i in range(ndat):
            if mono[i] > 0:
                offset += vel[i + ipre] * mono[i]
                count += mono[i]

        if count > 0:
            offset /= count

            # 计算校正误差
            for i in range(ndat):
                if mono[i] > 0:
                    err[i + ipre] = vel[i + ipre] - offset
                    rbserr += err[i + ipre] * err[i + ipre] * mono[i]

            rbserr = np.sqrt(rbserr / count)
        else:
            offset = 0.0
            rbserr = 0.0
    else:
        # 如果没有显著的速度变化
        offset = np.mean(vel[ipre : isdw + 1])
        rbserr = np.std(vel[ipre : isdw + 1])

    # 应用基线校正
    err[:nwin] = vel[:nwin] - offset

    return offset, rbserr
