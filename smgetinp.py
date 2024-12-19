import numpy as np
import os
from disazi import disazi
from smalloc import Constants, GlobalVars, AllocatableVars
from skipdoc import skipdoc


def smgetinp(input_file: str):
    """
    读取输入文件和地震数据信息

    参数:
        input_file: 输入文件路径

    返回:
        Tuple[Constants, GlobalVars, AllocatableVars, bool]: (常量, 全局变量, 可分配变量, 成功标志)
    """
    const = Constants()
    gv = GlobalVars()
    av = AllocatableVars()

    gv.inputfile = input_file

    print(" 正在读取输入文件...")

    try:
        # 读取主输入文件
        with open(input_file, "r") as f:
            # 读取地震发生时间
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取地震时间参数")
            year, month, day, hour, minute, gv.hyptime = map(float, line.split())
            gv.year, gv.month, gv.day = int(year), int(month), int(day)
            gv.hour, gv.minute = int(hour), int(minute)

            # 读取震源位置
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取震源位置")
            gv.hyplat, gv.hyplon, gv.hypdep = map(float, line.split())
            gv.hypdep *= const.KM2M  # 将深度从km转换为m

            # 读取数据目录
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取数据目录")
            gv.datadir = line.strip().strip("'\"")

            # 读取距离范围
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取距离范围")
            gv.stdismin, gv.stdismax = map(float, line.split())
            gv.stdismin *= const.KM2M
            gv.stdismax *= const.KM2M

            # 读取输出目录
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取输出目录")
            gv.outdir = line.strip().strip("'\"")
            if not os.path.exists(gv.outdir):
                os.makedirs(gv.outdir)
            gv.outdirlen = len(gv.outdir.rstrip("/\\"))

            # 读取coseis文件名
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取coseis文件名")
            gv.coseis = line.strip().strip("'\"")

            # 读取dt值
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取dt值")
            gv.dt = float(line)

            gv.coseis = os.path.join(gv.outdir, gv.coseis)
            gv.datadirlen = len(gv.datadir.rstrip("/\\"))

        # 读取SMDataInfo.dat文件
        sminfo_path = os.path.join(gv.datadir, "SMDataInfo.dat")
        with open(sminfo_path, "r") as f:
            # 验证地震参数
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取参考时间")
            year0, month0, day0, hour0, minute0, hyptime0 = map(float, line.split())

            if (
                int(year0) != gv.year
                or int(month0) != gv.month
                or int(day0) != gv.day
                or int(hour0) != gv.hour
                or int(minute0) != gv.minute
                or hyptime0 != gv.hyptime
            ):
                raise ValueError("地震发生时间不一致!")

            # 验证震源位置
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取参考位置")
            hyplat0, hyplon0, hypdep0 = map(float, line.split())
            hypdep0 *= const.KM2M
            if (
                hyplat0 != gv.hyplat
                or hyplon0 != gv.hyplon
                or abs(hypdep0 - gv.hypdep) > 1e-10
            ):
                raise ValueError("震源位置不一致!")

            # 读取台站数量和单位
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取台站数量")
            gv.nst, gv.accunit = int(line.split()[0]), float(line.split()[1])

            # 读取分量信息
            line = skipdoc(f)
            if not line:
                raise ValueError("无法读取分量信息")
            gv.icmp = list(map(int, line.split()))[:3]

            # 初始化可分配变量的数组
            av.stcode = []
            av.stclen = np.zeros(gv.nst, dtype=np.int32)
            av.lat = np.zeros(gv.nst, dtype=np.float64)
            av.lon = np.zeros(gv.nst, dtype=np.float64)
            av.start = np.zeros(gv.nst, dtype=np.float64)
            av.ponset = np.zeros(gv.nst, dtype=np.float64)
            av.length = np.zeros(gv.nst, dtype=np.float64)
            av.epidis = np.zeros(gv.nst, dtype=np.float64)
            av.sample = np.zeros(gv.nst, dtype=np.float64)
            av.offset = np.zeros((3, gv.nst), dtype=np.float64)
            av.rbserr = np.zeros((3, gv.nst), dtype=np.float64)
            av.tpga = np.zeros(gv.nst, dtype=np.float64)
            av.tsdw = np.zeros(gv.nst, dtype=np.float64)
            av.tddw = np.zeros(gv.nst, dtype=np.float64)
            av.okay = np.zeros(gv.nst, dtype=np.bool_)

            # 读取台站数据
            valid_stations = 0
            for ist in range(gv.nst):
                line = skipdoc(f)
                if not line:
                    raise ValueError(f"无法读取台站 {ist+1} 的数据")
                parts = line.split()
                stcode = parts[0]
                lat, lon = float(parts[1]), float(parts[2])
                start, ponset = float(parts[3]), float(parts[4])
                length, sample = float(parts[5]), float(parts[6])

                if sample <= 0:
                    raise ValueError(f"台站 {stcode} 的采样间隔无效")

                if ponset < start + const.PREWIN:
                    print(f"{stcode} ... 预震窗口时间不足 ...")
                    continue

                # 计算震中距
                dnorth, deast = disazi(const.REARTH, gv.hyplat, gv.hyplon, lat, lon)
                epidis = np.sqrt(dnorth**2 + deast**2)

                # 检查台站是否在距离范围内
                if (
                    epidis >= gv.stdismin
                    and epidis <= gv.stdismax
                    and ponset >= start + const.PREWIN
                ):
                    av.stcode.append(stcode)
                    av.lat[valid_stations] = lat
                    av.lon[valid_stations] = lon
                    av.start[valid_stations] = start
                    av.ponset[valid_stations] = ponset
                    av.length[valid_stations] = length
                    av.sample[valid_stations] = sample
                    av.epidis[valid_stations] = epidis
                    valid_stations += 1

            # 更新有效台站数量
            gv.nst = valid_stations
            if gv.nst <= 0:
                raise ValueError("没有可用的数据!")

            # 裁剪数组至实际大小
            av.stclen = av.stclen[: gv.nst]
            av.lat = av.lat[: gv.nst]
            av.lon = av.lon[: gv.nst]
            av.start = av.start[: gv.nst]
            av.ponset = av.ponset[: gv.nst]
            av.length = av.length[: gv.nst]
            av.epidis = av.epidis[: gv.nst]
            av.sample = av.sample[: gv.nst]
            av.offset = av.offset[:, : gv.nst]
            av.rbserr = av.rbserr[:, : gv.nst]
            av.tpga = av.tpga[: gv.nst]
            av.tsdw = av.tsdw[: gv.nst]
            av.tddw = av.tddw[: gv.nst]
            av.okay = av.okay[: gv.nst]

            # 按震中距排序台站
            sort_idx = np.argsort(av.epidis)
            av.stcode = [av.stcode[i] for i in sort_idx]
            av.stclen = av.stclen[sort_idx]
            av.lat = av.lat[sort_idx]
            av.lon = av.lon[sort_idx]
            av.start = av.start[sort_idx]
            av.ponset = av.ponset[sort_idx]
            av.length = av.length[sort_idx]
            av.epidis = av.epidis[sort_idx]
            av.sample = av.sample[sort_idx]
            av.offset = av.offset[:, sort_idx]
            av.rbserr = av.rbserr[:, sort_idx]
            av.tpga = av.tpga[sort_idx]
            av.tsdw = av.tsdw[sort_idx]
            av.tddw = av.tddw[sort_idx]
            av.okay = av.okay[sort_idx]

            # 计算最大窗口大小和台站代码长度
            gv.nwinmax = max(
                1 + 2 * int(length / sample)
                for length, sample in zip(av.length, av.sample)
            )
            av.stclen = np.array(
                [len(code.strip()) for code in av.stcode], dtype=np.int32
            )

            # 分配时间序列数据数组
            av.acc = np.zeros((gv.nwinmax, 3), dtype=np.float64)
            av.vel = np.zeros((gv.nwinmax, 3), dtype=np.float64)
            av.dis = np.zeros((gv.nwinmax, 3), dtype=np.float64)
            av.err = np.zeros((gv.nwinmax, 3), dtype=np.float64)
            av.dat = np.zeros((gv.nwinmax, 3), dtype=np.float64)
            av.swp = np.zeros(gv.nwinmax, dtype=np.float64)
            av.ene = np.zeros(gv.nwinmax, dtype=np.float64)

        print(" 成功读取输入参数")
        print(f" 数据目录: {gv.datadir}")
        print(f" 输出目录: {gv.outdir}")
        print(f" 台站数量: {gv.nst}")
        print(
            f" 距离范围: {gv.stdismin/const.KM2M:.1f} - {gv.stdismax/const.KM2M:.1f} km"
        )
        print(f" 采样间隔: {gv.dt:.3f} s")

        return const, gv, av, True

    except Exception as e:
        print(f" smgetinp出错: {str(e)}")
        return const, gv, av, False
