# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import Ideal_Filter,Butterworth_Filter,Gaussian_Filter

if __name__=='__main__':
    image_path='./image/Cameraman.bmp' #要读取的图片的文件路径
    image_mat=np.array(Image.open(image_path).convert('L')) #获取到图片的矩阵表示
    M,N=image_mat.shape[0],image_mat.shape[1] #图片的维度

    P,Q=2*M,2*N #0-padding
    fp=np.zeros((P,Q))

    #对f(x,y)添加必要数量的0,形成大小为P*Q的填充后的图像fp
    for i in range(M):
        for j in range(N):
            fp[i][j]=image_mat[i][j]

    #用(-1)^(i+j)乘以fp(i,j)将其移到变换的中心
    Fp=np.zeros((P,Q))
    for i in range(P):
        for j in range(Q):
            Fp[i][j]=math.pow(-1,i+j)*fp[i][j]

    #对图像进行DFT
    F=np.fft.fft2(Fp)
    F_pu=np.log(np.abs(F))

    D0=100 #截止频率

    # 对DFT后的F与滤波函数进行阵列相乘
    #G1=Ideal_Filter.ILPF(P,Q,D0)*F #与理想低通滤波器作用
    #G2=Ideal_Filter.IHPF(P,Q,D0)*F #与理想高通滤波器作用

    #与布特沃斯滤波器作用
    #G1=Butterworth_Filter.BLPF(P,Q,D0,2)*F
    #G2=Butterworth_Filter.BHPF(P,Q,D0,2)*F

    #与高斯滤波器作用
    G1=Gaussian_Filter.Gaussian_lowpass_filter(P,Q,D0)*F
    G2=Gaussian_Filter.Gaussian_highpass_filter(P,Q,D0)*F
    G1_pu = np.log(np.abs(G1)) #G1的谱
    #G2_pu = np.log(np.abs(G2)) #G2的谱

    #反中心化
    G1=np.fft.ifftshift(G1)
    G2=np.fft.ifftshift(G2)

    #逆傅里叶变换并取实数部分
    gp1=np.fft.ifft2(G1)
    gp1=np.real(gp1)
    gp2=np.fft.ifft2(G2)
    gp2=np.real(gp2)

    #截取原图的尺寸作为输出
    g1=np.zeros((M,N))
    g2=np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            g1[i][j]=gp1[i][j]
            g2[i][j]=gp2[i][j]

    #绘制对比图像
    plt.subplot(2,4,1)
    plt.imshow(image_mat,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,2)
    plt.imshow(fp,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,3)
    plt.imshow(Fp,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,4)
    plt.imshow(F_pu,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,5)
    plt.imshow(Gaussian_Filter.Gaussian_lowpass_filter(P,Q,D0),cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,6)
    plt.imshow(G1_pu,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,7)
    plt.imshow(gp1,cmap='gray')
    plt.axis('off')

    plt.subplot(2,4,8)
    plt.imshow(g1,cmap='gray')
    plt.axis('off')
    plt.savefig('process_Cameraman_Gaussian_lowpass.png',dpi=512)
    plt.show()
