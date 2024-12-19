def skipdoc(file_handle):
    """
    跳过文件中的注释行和空行

    参数:
    file_handle: 文件句柄对象

    返回:
    str: 第一个非注释非空行，如果到达文件末尾则返回None
    """
    for line in file_handle:
        line = line.strip()
        if line and not line.startswith(("#", "!", "c", "C")):
            return line
    return None
