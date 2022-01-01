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


def gaussian_func(distance,sigma):
    '''
    高斯距离函数, 由像素点之间的distance和sigma值根据高斯函数度量
    :param distance: 两个像素点之间的距离, type=float
    :param sigma: 高斯滤波算法的sigma值, type=float
    :return: 高斯距离值, type=float
    '''
    return np.exp(-0.5*np.square(distance/sigma))/(2*np.pi*np.square(sigma))


def gaussian_filter(init_image_path,kernel_size,sigma):
    '''
    高斯滤波器
    :param init_image: 原始图像路径, type=string
    :param kernel_size: filter大小, type=list
    :param sigma: sigma的值, type=float
    :return: 滤波操作后的图像, type=Image
    '''
    init_image = Image.open(init_image_path).convert('L')  # 用PIL中的Image.open打开图像并转化为灰度图
    init_image_mat = np.array(init_image)  # 将原始图像转化成numpy数组
    filtered_image_mat=np.zeros_like(init_image_mat)
    #对图像的每一个像素值进行高斯滤波操作
    for x in range(init_image_mat.shape[0]):
        for y in range(init_image_mat.shape[1]):
            print("finish x=%d,y=%d"%(x,y))
            filtered_pixel=0
            weight_sum=0
            #每个像素的操作范围为以其为中心的kernel
            for i in range(-(kernel_size[0]//2),(kernel_size[0]//2)+1):
                for j in range(-(kernel_size[1]//2),(kernel_size[1]//2)+1):
                    #遍历周边滤波模板内的中心点的周边像素点
                    now_x=x+i
                    now_y=y+j
                    #特判超出边界的不合法坐标
                    if now_x<0 or now_x>=init_image_mat.shape[0] or now_y<0 or now_y>=init_image_mat.shape[1]:
                        continue
                    dis=euclidean_distance(x,y,now_x,now_y) #距离
                    weight=gaussian_func(dis,sigma) #高斯权重度量
                    filtered_pixel+=weight*init_image_mat[now_x][now_y]
                    weight_sum+=weight
            filtered_image_mat[x][y]=int(round(filtered_pixel/weight_sum))
    filtered_image=Image.fromarray(filtered_image_mat)

    return filtered_image


if __name__=='__main__':
    # test
    init_image_path="./images/Cameraman_noise.bmp"
    kernel_size=[5,5]
    sigma=5
    filtered_image=gaussian_filter(init_image_path,kernel_size,sigma)
    filtered_image.show()
