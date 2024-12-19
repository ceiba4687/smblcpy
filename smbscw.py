import numpy as np
from linefit import linefit
from bscmono import bscmono
from smalloc import (
    GlobalVars,
    AllocatableVars,
    Constants,
)


def smbscw(
    ist: int, nwin: int, gv: GlobalVars, av: AllocatableVars, const: Constants
) -> int:
    """
    地震波形基线校正函数 (Strong Motion Baseline Correction Window)

    参数说明:
    ist: int, 台站索引
    nwin: int, 时间窗口长度
    glob: GlobalVars, 全局变量
    alloc: AllocatableVars, 可分配变量

    返回值:
    nwin: int, 更新后的时间窗口长度
    """
    # 选择预事件时间窗口
    ipre = 1 + int((av.ponset[ist] - av.start[ist] - const.DTP) / gv.dt)
    k = ipre - 1 - int(6.0 * const.PREWIN / gv.dt)

    if k > 0:
        ipre -= k
        nwin -= k
        av.start[ist] += float(k) * gv.dt
        av.length[ist] -= float(k) * gv.dt
        av.acc[:nwin] = av.acc[k : k + nwin]

    # 移除预事件静态偏移
    for j in range(3):
        preoff = np.mean(av.acc[:ipre, j])
        av.acc[:nwin, j] -= preoff

    # 选择信号和后事件时间窗口
    ipga = 1 + int((av.tpga[ist] - av.start[ist]) / gv.dt)
    isdw = 1 + int((av.tsdw[ist] - av.start[ist]) / gv.dt)
    iddw = 1 + int((av.tddw[ist] - av.start[ist]) / gv.dt)

    # 更新时间窗口长度
    nwin = min(nwin, iddw + isdw - ipre)

    # 对加速度记录进行积分并校正基线误差
    for j in range(3):
        # 计算未校正的速度
        av.vel[0, j] = 0.0
        for i in range(1, nwin):
            av.vel[i, j] = av.vel[i - 1, j] + av.acc[i, j] * gv.dt

        # 拟合预事件基线
        al, bl = linefit(ipre, av.vel[:ipre, j])

        # 更新预事件基线校正
        t = np.arange(nwin, dtype=np.float64)
        av.vel[:nwin, j] -= al + (bl - al) * t / float(ipre - 1)
        av.acc[:nwin, j] -= (bl - al) / (float(ipre - 1) * gv.dt)

        # 寻找最大加速度
        ipgaj = ipre
        pgaj = 0.0
        for i in range(ipre, isdw):
            if abs(av.acc[i, j]) > pgaj:
                pgaj = abs(av.acc[i, j])
                ipgaj = i

        # 进行单调基线校正
        av.offset[j, ist], av.rbserr[j, ist] = bscmono(
            nwin,
            ipre,
            min(ipga, ipgaj),
            isdw,
            av.vel[:nwin, j],
            av.err[:nwin, j],
            gv.dt,
        )

        # 计算位移
        av.dis[0, j] = 0.0
        for i in range(1, nwin):
            av.dis[i, j] = av.dis[i - 1, j] + av.vel[i, j] * gv.dt

    return nwin
