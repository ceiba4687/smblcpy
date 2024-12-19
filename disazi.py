import numpy as np

def disazi(rearth, lateq, loneq, latst, lonst):
    """
    计算球面上两点之间的距离和方位角
    
    参数:
    rearth: 地球半径
    lateq: 震中纬度(度)
    loneq: 震中经度(度)
    latst: 台站纬度(度)
    lonst: 台站经度(度)
    
    返回:
    xnorth, yeast: 台站相对于震中的北向和东向距离(km)
    """
    # 常量定义
    PI = np.pi
    PI2 = 2.0 * PI
    DEGTORAD = PI / 180.0
    
    # 将经纬度转换为弧度
    latb = lateq * DEGTORAD  # 震中纬度
    lonb = loneq * DEGTORAD  # 震中经度
    latc = latst * DEGTORAD  # 台站纬度
    lonc = lonst * DEGTORAD  # 台站经度
    
    # 处理经度范围
    if lonb < 0.0:
        lonb += PI2
    if lonc < 0.0:
        lonc += PI2
    
    # 计算球面三角形的边
    b = 0.5 * PI - latb
    c = 0.5 * PI - latc
    
    # 确定经度差角度
    if lonc > lonb:
        aa = lonc - lonb
        if aa <= PI:
            iangle = 1
        else:
            aa = PI2 - aa
            iangle = -1
    else:
        aa = lonb - lonc
        if aa <= PI:
            iangle = -1
        else:
            aa = PI2 - aa
            iangle = 1
    
    # 计算球面距离
    s = np.cos(b) * np.cos(c) + np.sin(b) * np.sin(c) * np.cos(aa)
    # 使用np.clip避免数值误差导致的域错误
    s = np.clip(s, -1.0, 1.0)
    a = np.arccos(s)
    dis = a * rearth
    
    # 计算方位角
    if a * b * c == 0.0:
        angleb = 0.0
        anglec = 0.0
    else:
        s = 0.5 * (a + b + c)
        a = min(a, s)
        b = min(b, s)
        c = min(c, s)
        
        # 计算球面三角形的角度
        sin_term_c = np.sin(s-a) * np.sin(s-b) / (np.sin(a) * np.sin(b))
        sin_term_b = np.sin(s-a) * np.sin(s-c) / (np.sin(a) * np.sin(c))
        
        # 使用np.clip避免数值误差
        anglec = 2.0 * np.arcsin(np.clip(np.sqrt(sin_term_c), 0.0, 1.0))
        angleb = 2.0 * np.arcsin(np.clip(np.sqrt(sin_term_b), 0.0, 1.0))
        
        if iangle == 1:
            angleb = PI2 - angleb
        else:
            anglec = PI2 - anglec
    
    # 计算直角坐标
    xnorth = dis * np.cos(anglec)
    yeast = dis * np.sin(anglec)
    
    return xnorth, yeast