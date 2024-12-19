import numpy as np
import os
from smalloc import Constants, GlobalVars, AllocatableVars
from skipdoc import skipdoc
from smbscw import smbscw


def smgetout(const: Constants, gv: GlobalVars, av: AllocatableVars) -> bool:
    """
    读取强震动数据并进行基线校正

    参数:
        const: Constants实例，包含常量
        gv: GlobalVars实例，包含全局变量
        av: AllocatableVars实例，包含可分配变量

    返回:
        bool: 是否成功
    """
    print(" 读取强震动数据...")
    print(" 进行基线校正...")

    print(
        "   Station  Lat[deg]  Lon[deg] Epdis[km]   East[m]  North[m]     Up[m]"
        "   RbserrE   RbserrN   RbserrU"
    )

    try:
        for ist in range(gv.nst):
            # 读取强震动数据
            data_file = os.path.join(gv.datadir, f"{av.stcode[ist]}.dat")
            data_list = []
            with open(data_file, "r") as f:
                for line in f:
                    values = list(map(float, line.split()))
                    if len(values) >= 3:
                        data_list.append([values[i - 1] for i in gv.icmp])
                    if len(data_list) >= gv.nwinmax:
                        break

            nwin = len(data_list)
            av.length[ist] = (nwin - 1) * av.sample[ist]
            dat = np.array(data_list)

            # 初始地震前基线校正
            ipre = 1 + int(
                (av.ponset[ist] - av.start[ist] - const.DTP) / av.sample[ist]
            )
            accoff = np.zeros(3)

            for j in range(3):
                delta = np.mean(dat[:ipre, j])
                dat[:, j] -= delta
                accoff[j] = np.mean(dat[ipre:, j])

            # 确定PGA时间和地震后时期的开始
            ene = np.zeros(nwin)
            sigma = np.sqrt(np.sum((dat[ipre:] - accoff) ** 2, axis=1))
            ipga = ipre + np.argmax(sigma)
            pga = np.max(sigma)

            ene[ipre:] = np.cumsum(sigma)
            av.tpga[ist] = av.start[ist] + ipga * av.sample[ist]

            nwin = min(
                nwin,
                ipre + 20 * round((av.tpga[ist] - av.ponset[ist]) / av.sample[ist]),
            )

            # 确定SDW和DDW时间
            isdw = ipre + np.searchsorted(ene[ipre:nwin], const.SDW * ene[nwin - 1])
            iddw = ipre + np.searchsorted(ene[ipre:nwin], const.DDW * ene[nwin - 1])

            av.tsdw[ist] = av.start[ist] + isdw * av.sample[ist]
            av.tddw[ist] = av.start[ist] + iddw * av.sample[ist]

            if av.tsdw[ist] < av.tpga[ist]:
                av.tpga[ist] = av.tsdw[ist]

            # 调整长度
            av.length[ist] = min(
                av.length[ist],
                av.tddw[ist]
                - av.start[ist]
                + min(av.tpga[ist] - av.start[ist], const.PSTWIN),
            )
            nwin = int(av.length[ist] / av.sample[ist])

            # 检查数据是否足够
            av.okay[ist] = av.tsdw[ist] <= av.start[ist] + av.length[ist] - min(
                av.tpga[ist] - av.start[ist], const.PSTWIN
            )

            if not av.okay[ist]:
                print(f"{av.stcode[ist]}   ... 数据长度不足 ...")
                continue

            # 降采样
            if av.sample[ist] < gv.dt:
                nsam = round(gv.dt / av.sample[ist])
            else:
                nsam = 1

            ipre = 1 + int((av.ponset[ist] - av.start[ist] - const.DTP) / gv.dt)
            nwin = nwin // nsam

            # 进行降采样
            for i in range(nwin):
                l = max(0, int((i - 0.5) * gv.dt / av.sample[ist]))
                for j in range(3):
                    av.acc[i, j] = gv.accunit * np.mean(dat[l : l + nsam, j])

            av.sample[ist] = gv.dt

            # 进行基线校正

            smbscw(ist, nwin, gv, av, const)

            # 输出校正结果
            print(
                f"{av.stcode[ist]:10} {av.lat[ist]:8.4f} {av.lon[ist]:8.4f}"
                f" {av.epidis[ist]/const.KM2M:8.3f}"
                f" {av.offset[0,ist]:8.3f} {av.offset[1,ist]:8.3f}"
                f" {av.offset[2,ist]:8.3f}"
                f" {av.rbserr[0,ist]:8.4f} {av.rbserr[1,ist]:8.4f}"
                f" {av.rbserr[2,ist]:8.4f}"
            )

            # 保存校正后的数据
            outfile = os.path.join(gv.outdir, f"{av.stcode[ist]}_blc.dat")
            with open(outfile, "w") as f:
                f.write(
                    "        Time          VdatE          VdatN          VdatZ"
                    "         BlerrE         BlerrN         BlerrZ"
                    "      VelocityE      VelocityN      VelocityZ"
                    "  DisplacementE  DisplacementN  DisplacementZ\n"
                )

                for i in range(nwin):
                    time = av.start[ist] + i * gv.dt
                    vdat = av.vel[i] + av.err[i]
                    f.write(f"{time:12.3f}")
                    f.write("".join(f"{v:15.7E}" for v in vdat))
                    f.write("".join(f"{e:15.7E}" for e in av.err[i]))
                    f.write("".join(f"{v:15.7E}" for v in av.vel[i]))
                    f.write("".join(f"{d:15.7E}" for d in av.dis[i]))
                    f.write("\n")

        # 保存同震位移结果
        with open(gv.coseis, "w") as f:
            f.write(
                "   Station  Lat[deg]  Lon[deg] Epdis[km]   East[m]  "
                "North[m]     Up[m]   RbserrE   RbserrN   RbserrU\n"
            )

            valid_stations = 0
            for ist in range(gv.nst):
                if av.okay[ist]:
                    valid_stations += 1
                    f.write(
                        f"{av.stcode[ist]:10} {av.lat[ist]:8.4f}"
                        f" {av.lon[ist]:8.4f}"
                        f" {av.epidis[ist]/const.KM2M:8.3f}"
                        f" {av.offset[0,ist]:8.3f} {av.offset[1,ist]:8.3f}"
                        f" {av.offset[2,ist]:8.3f}"
                        f" {av.rbserr[0,ist]:8.4f} {av.rbserr[1,ist]:8.4f}"
                        f" {av.rbserr[2,ist]:8.4f}\n"
                    )

        gv.nst = valid_stations
        print(f" ====== {gv.nst}个台站的基线校正完成 =======")
        return True

    except Exception as e:
        print(f" smgetout出错: {str(e)}")
        return False
