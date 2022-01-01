# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def euclidean_distance(x1,y1,x2,y2):
    '''
    计算两个像素点之间的欧拉距离
    :param (x1,y1):像素点1的坐标, type=int
    :param (x2,y2):像素点2的坐标, type=int
    :return:两个像素点之间的欧拉距离, type=float
    '''
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


def pixel_difference(p1,p2):
    '''
    计算两个像素点之间的灰度值距离
    :param p1: 像素点1的灰度值, type=int
    :param p2: 像素点2的灰度值, type=int
    :return: 两个像素点之间的灰度值距离, type=int
    '''
    return abs(int(p1)-int(p2))


def closeness_func(distance,sigma_d):
    '''
    接近程度函数, 对两个像素点的距离根据高斯函数求得度量双边滤波算法权重的第一项值
    :param distance: 两个像素点之间的距离, type=float
    :param sigma_d: 双边滤波算法的sigma元组的第一项sigma值, type=float
    :return: 像素点之间接近程度, 即双边滤波算法权重的第一项值, type=float
    '''
    c=np.exp(-0.5*np.square(distance/sigma_d))
    return c


def similarity_func(difference,sigma_r):
    '''
    灰度相似程度函数, 对两个像素点的灰度差根据高数函数形式变换求得度量双边滤波算法权重的第二项值
    :param difference: 两个像素点之间的灰度差, type=float
    :param sigma_r: 双边滤波算法的sigma元组的第二项sigma值, type=float
    :return: 灰度接近程度, 即双边滤波算法权重的第二项值, type=float
    '''
    s=np.exp(-0.5*np.square(difference/sigma_r))
    return s


def multi_size_bilateral_filtering(init_image_path,multi_kernel_size,sigma):
    '''
    多粒度双边滤波操作
    :param init_image: 原始图像路径, type=string
    :param multi_kernel_size: filter大小, type=list
    :param sigma: sigma_d和sigma_r的值, type=list
    :return: 滤波操作后的图像, type=Image
    '''
    init_image = Image.open(init_image_path).convert('L')  # 用PIL中的Image.open打开图像并转化为灰度图
    init_image_mat = np.array(init_image)  # 将原始图像转化成numpy数组
    sigma_d=sigma[0]
    sigma_r=sigma[1]
    filtered_image_mat=np.zeros_like(init_image_mat,dtype=int)
    #对图像的每一个像素值进行双边滤波操作
    #由多粒度的滤波器分别进行滤波操作
    for kernel in range(multi_kernel_size.shape[0]):
        for x in range(init_image_mat.shape[0]):
            for y in range(init_image_mat.shape[1]):
                print("finish x=%d, y=%d"%(x,y))
                filtered_pixel=0
                weight_sum=0
                # 每个像素的操作范围为以其为中心的kernel
                for i in range(-(multi_kernel_size[kernel][0]//2),(multi_kernel_size[kernel][0]//2)+1):
                    for j in range(-(multi_kernel_size[kernel][1]//2),(multi_kernel_size[kernel][1]//2)+1):
                        # 遍历周边滤波模板内的中心点的周边像素点
                        now_x=x+i
                        now_y=y+j
                        # 特判超出边界的不合法坐标
                        if now_x<0 or now_x>=init_image_mat.shape[0] or now_y<0 or now_y>=init_image_mat.shape[1]:
                            continue
                        dis=euclidean_distance(x,y,now_x,now_y) #距离
                        diff=pixel_difference(init_image_mat[x][y],init_image_mat[now_x][now_y]) #像素差
                        closeness=closeness_func(dis,sigma_d) #双边滤波函数的第一项权重值
                        similarity=similarity_func(diff,sigma_r) #双边滤波函数的第二项权重值
                        weight=closeness*similarity #双边滤波函数的权重: 第一项*第二项
                        filtered_pixel+=weight*init_image_mat[now_x][now_y] #按权重聚合周边像素点值
                        weight_sum+=weight #记录总权重,用于标准化
                #标准化及更新
                filtered_image_mat[x][y]+=int(round(filtered_pixel/weight_sum))

    #对进行的多个粒度的双边滤波结果取均值
    for i in range(filtered_image_mat.shape[0]):
        for j in range(filtered_image_mat.shape[1]):
            filtered_image_mat[i][j]=int(round(filtered_image_mat[i][j]/multi_kernel_size.shape[0]))

    #绘制对比图像
    plt.subplot(1,2,1)
    plt.title('origin')
    plt.imshow(init_image_mat,cmap="gray")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('filtered')
    plt.imshow(filtered_image_mat,cmap="gray")
    plt.axis('off')
    plt.show()

    return filtered_image_mat


if __name__=='__main__':
    # test
    init_image_path="./images/Cameraman_noise.bmp"
    multi_kernel_size=np.array([[5,5],[3,3]])
    sigma=[10,10]
    filtered_image_mat=multi_size_bilateral_filtering(init_image_path,multi_kernel_size,sigma)
