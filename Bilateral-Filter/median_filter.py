# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def median_filter(init_image_path,kernel_size):
    '''
    线性中值滤波器
    :param init_image: 原始图像路径, type=string
    :param kernel_size: filter大小, type=list
    :return: 滤波操作后的图像, type=Image
    '''
    init_image = Image.open(init_image_path).convert('L')  # 用PIL中的Image.open打开图像并转化为灰度图
    init_image_mat = np.array(init_image)  # 将原始图像转化成numpy数组
    filtered_image_mat=np.zeros_like(init_image_mat)
    #对图像的每一个像素值进行线性中值滤波操作
    for x in range(init_image_mat.shape[0]):
        for y in range(init_image_mat.shape[1]):
            print("finish x=%d,y=%d"%(x,y))
            #每个像素的操作范围为以其为中心的kernel
            weight_list=[]
            for i in range(-(kernel_size[0]//2),(kernel_size[0]//2)+1):
                for j in range(-(kernel_size[1]//2),(kernel_size[1]//2)+1):
                    now_x=x+i
                    now_y=y+j
                    #特判超出边界的不合法坐标
                    if now_x<0 or now_x>=init_image_mat.shape[0] or now_y<0 or now_y>=init_image_mat.shape[1]:
                        continue
                    weight_list.append(init_image_mat[now_x][now_y])
            #标准化及更新
            weight_list.sort()
            filtered_image_mat[x][y]=weight_list[len(weight_list)//2]
    filtered_image=Image.fromarray(filtered_image_mat)

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

    return filtered_image


if __name__=='__main__':
    # test
    init_image_path="./images/Cameraman_noise.bmp"
    kernel_size=[3,3]
    filtered_image=median_filter(init_image_path,kernel_size)
