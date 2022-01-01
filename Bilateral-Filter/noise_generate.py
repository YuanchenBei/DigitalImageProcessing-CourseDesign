# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


def gauss_noise(image, mean=0, var=0.0005):
    '''
    为图像添加高斯噪声
    :param image: 要添加噪声的图像, type=Image
    :param mean: 均值
    :param var: 方差, 方差越大所加噪声越大
    :return: 加噪声后的图像
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out


if __name__=='__main__':
    # test
    ori_image=Image.open("./images/Peppers.bmp")
    ori_image_mat=np.array(ori_image)
    noise_image_mat=gauss_noise(ori_image_mat)
    noise_image = Image.fromarray(noise_image_mat)
    noise_image.show()
    noise_image.save('./images/Peppers_noise.bmp')
