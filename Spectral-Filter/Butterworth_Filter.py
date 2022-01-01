# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def BLPF(P,Q,D0=100,n=2):
    '''
    布特沃斯低通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :param n: 布特沃斯滤波器的阶数, default value=2
    :return: 布特沃斯低通滤波器, type=np.array
    '''
    D=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2))

    BLPF_mat=np.zeros((P,Q)) #布特沃斯低通滤波器
    for i in range(P):
        for j in range(Q):
            BLPF_mat[i][j]=1/(1+math.pow(D[i][j]/D0,2*n))

    return BLPF_mat


def BHPF(P,Q,D0=100,n=2):
    '''
    布特沃斯高通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :param n: 布特沃斯滤波器的阶数, default value=2
    :return: 布特沃斯低通滤波器, type=np.array
    '''
    D=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2))

    BHPF_mat=np.zeros((P,Q)) #布特沃斯高通滤波器
    for i in range(P):
        for j in range(Q):
            BHPF_mat[i][j]=1-(1/(1+math.pow(D[i][j]/D0,2*n)))

    return BHPF_mat