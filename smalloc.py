"""
地震数据管理模块 (对应原Fortran smalloc模块)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, List
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Constants:
    """全局常量 (对应原Fortran parameter部分)"""

    REARTH: Final[float] = 6.371e6  # 地球半径(m)
    KM2M: Final[float] = 1.0e3  # 千米到米的转换系数
    DEG2RAD: Final[float] = 0.01745329251994328  # 角度到弧度的转换系数
    PREWIN: Final[float] = 5.0  # 预事件窗口(s)
    PSTWIN: Final[float] = 25.0  # 后事件窗口(s)
    DTP: Final[float] = 1.5  # P波到时容差(s)
    SDW: Final[float] = 0.85  # S波权重
    DDW: Final[float] = 0.95  # 直达波权重


class GlobalVars:
    """全局变量 (对应原Fortran全局变量部分)"""

    def __init__(self):
        # 整型变量
        self.nwinmax: int = 0  # 最大窗口数
        self.nst: int = 0  # 台站数量
        self.year: int = 0  # 年
        self.month: int = 0  # 月
        self.day: int = 0  # 日
        self.hour: int = 0  # 时
        self.minute: int = 0  # 分
        self.datadirlen: int = 0  # 数据目录长度
        self.outdirlen: int = 0  # 输出目录长度
        self.icmp: List[int] = [0] * 3  # 分量标识

        # 浮点数变量
        self.hyptime: float = 0.0  # 发震时刻
        self.hyptime0: float = 0.0  # 参考发震时刻
        self.hyplat: float = 0.0  # 震源纬度
        self.hyplon: float = 0.0  # 震源经度
        self.hypdep: float = 0.0  # 震源深度
        self.hyplat0: float = 0.0  # 参考震源纬度
        self.hyplon0: float = 0.0  # 参考震源经度
        self.hypdep0: float = 0.0  # 参考震源深度
        self.stdismin: float = 0.0  # 最小震中距
        self.stdismax: float = 0.0  # 最大震中距
        self.dt: float = 0.0  # 采样间隔
        self.accunit: float = 0.0  # 加速度单位

        # 字符串变量
        self.stswp: str = ""  # 10字符
        self.datadir: str = ""  # 80字符
        self.outdir: str = ""  # 80字符
        self.inputfile: str = ""  # 80字符
        self.coseis: str = ""  # 80字符


class AllocatableVars:
    """可分配变量 (对应原Fortran可分配变量部分)"""

    def __init__(self):
        # 一维数组
        self.stclen: npt.NDArray[np.int32] = np.array([], dtype=np.int32)
        self.lat: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.lon: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.start: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.ponset: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.epidis: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.tpga: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.tsdw: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.tddw: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.length: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.sample: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.swp: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.ene: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.okay: npt.NDArray[np.bool_] = np.array([], dtype=np.bool_)
        self.stcode: List[str] = []  # 10字符的字符串数组

        # 二维数组
        self.acc: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.vel: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.dis: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.err: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.dat: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.offset: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self.rbserr: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
