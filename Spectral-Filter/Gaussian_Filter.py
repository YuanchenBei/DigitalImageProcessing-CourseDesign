# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def Gaussian_lowpass_filter(P,Q,D0=100):
    '''
    高斯低通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :return: 高斯低通滤波器, type=np.array
    '''
    D=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2))

    Gaussian_lowpass=np.zeros((P,Q)) #高斯低通滤波器
    for i in range(P):
        for j in range(Q):
            Gaussian_lowpass[i][j]=np.exp(-1*math.pow(D[i][j],2)/(2*D0*D0))

    return Gaussian_lowpass


def Gaussian_highpass_filter(P,Q,D0=100):
    '''
    高斯高通滤波器
    :param P: 处理区域行数
    :param Q: 处理区域列数
    :param D0: 截止频率, default value=100
    :return: 高斯高通滤波器, type=np.array
    '''
    D=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            D[i][j]=math.sqrt(math.pow(i-P//2,2)+math.pow(j-Q//2,2))

    Gaussian_highpass=np.zeros((P,Q)) #高斯高通滤波器
    for i in range(P):
        for j in range(Q):
            Gaussian_highpass[i][j]=1-np.exp(-1*math.pow(D[i][j],2)/(2*D0*D0))

    return Gaussian_highpass
