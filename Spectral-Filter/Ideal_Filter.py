# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def ILPF(P,Q,D0=100):
    '''
    理想低通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :return: 理想低通滤波器, type=np.array
    '''
    D=np.zeros((P,Q)) #距离矩阵
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2)) #计算各像素点到中心的距离

    ILPF_mat=np.zeros((P,Q)) #理想低通滤波器的计算式
    for i in range(P):
        for j in range(Q):
            if D[i][j]<=D0:
                ILPF_mat[i][j]=1
            else:
                ILPF_mat[i][j]=0
    return ILPF_mat


def IHPF(P,Q,D0=100):
    '''
    理想高通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :return: 理想高通滤波器, type=np.array
    '''
    D=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2))

    IHPF_mat=np.zeros((P,Q)) #理想高通滤波器
    for i in range(P):
        for j in range(Q):
            if D[i][j]<=D0:
                IHPF_mat[i][j]=0
            else:
                IHPF_mat[i][j]=1
    return IHPF_mat
