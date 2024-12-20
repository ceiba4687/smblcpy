import sys
from smgetinp import smgetinp
from smgetout import smgetout
from smalloc import Constants, GlobalVars, AllocatableVars


def main():
    """
    主程序入口
    """
    try:
        # 获取输入文件名
        # if len(sys.argv) > 1:
        #     # 如果通过命令行参数提供输入文件
        #     input_file = sys.argv[1]
        # else:
        #     # 否则通过用户输入获取
        #     input_file = input("Input file name: ").strip()
        #     if not input_file:
        #         raise ValueError("No input file specified")
        input_file = "smblc20230206_turkey_M77.inp"
        # 读取数据
        print("Reading data...")
        const, gv, av, success = smgetinp(input_file)
        if not success:
            raise ValueError("Failed to read input file")

        # 进行基线校正
        print("Performing baseline correction...")
        smgetout(const, gv, av)

        print("Processing completed successfully")
        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
