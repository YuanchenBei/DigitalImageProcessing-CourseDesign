# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import bilateral_filter
import Gaussian_filter
import mean_filter
import median_filter
import multi_size_bilateral_filter
import noise_generate

#调用编写的各个函数模块,实现各个实验步骤
if __name__=='__main__':
    #1.双边滤波算法的实现
    #####################################################################
    origin_image_path="./images/Goldhill.bmp"
    init_image_path="./images/Goldhill_noise.bmp"
    kernel_size=[7,7]
    sigma=[15,15]
    filtered_image=bilateral_filter.bilateral_filtering_gray(init_image_path,kernel_size,sigma)

    origin_image_mat = np.array(Image.open(origin_image_path).convert('L'))
    init_image_mat = np.array(Image.open(init_image_path).convert('L'))

    #绘制对比图像
    plt.subplot(1,3,1)
    plt.title('original')
    plt.imshow(origin_image_mat,cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Gaussian noisy')
    plt.imshow(init_image_mat,cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title('bilateral filtered')
    plt.imshow(np.array(filtered_image),cmap='gray')
    plt.axis('off')
    plt.savefig('Goldhill_7_15_15.png',dpi=512)
    plt.show()
    #####################################################################

    '''
    #2.双边滤波算法的不同参数对处理效果的影响
    #####################################################################
    origin_image_path="./images/Peppers.bmp"
    kernel_size=[15,15]
    sigma11=[1,1]
    sigma12=[1,10]
    sigma13=[1,100]
    sigma14=[1,300]
    sigma21=[10,1]
    sigma22=[10,10]
    sigma23=[10,100]
    sigma24=[10,300]
    sigma31=[100,1]
    sigma32=[100,10]
    sigma33=[100,100]
    sigma34=[100,300]
    sigma41=[300,1]
    sigma42=[300,10]
    sigma43=[300,100]
    sigma44=[300,300]
    
    filtered_image11=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma11)
    filtered_image12=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma12)
    filtered_image13=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma13)
    filtered_image14=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma14)
    
    filtered_image21=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma21)
    filtered_image22=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma22)
    filtered_image23=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma23)
    filtered_image24=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma24)
    
    filtered_image31=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma31)
    filtered_image32=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma32)
    filtered_image33=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma33)
    filtered_image34=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma34)
    
    filtered_image41=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma41)
    filtered_image42=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma42)
    filtered_image43=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma43)
    filtered_image44=bilateral_filter.bilateral_filtering_gray(origin_image_path,kernel_size,sigma44)

    plt.subplot(4,4,1)
    plt.imshow(np.array(filtered_image11),cmap='gray')
    plt.axis('off')

    plt.subplot(4,4,2)
    plt.imshow(np.array(filtered_image12),cmap='gray')
    plt.axis('off')

    plt.subplot(4,4,3)
    plt.imshow(np.array(filtered_image13),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,4)
    plt.imshow(np.array(filtered_image14),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,5)
    plt.imshow(np.array(filtered_image21),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,6)
    plt.imshow(np.array(filtered_image22),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,7)
    plt.imshow(np.array(filtered_image23),cmap='gray')
    plt.axis('off')    
    
    plt.subplot(4,4,8)
    plt.imshow(np.array(filtered_image24),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,9)
    plt.imshow(np.array(filtered_image31),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,10)
    plt.imshow(np.array(filtered_image32),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,11)
    plt.imshow(np.array(filtered_image33),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,12)
    plt.imshow(np.array(filtered_image34),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,13)
    plt.imshow(np.array(filtered_image41),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,14)
    plt.imshow(np.array(filtered_image42),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,15)
    plt.imshow(np.array(filtered_image43),cmap='gray')
    plt.axis('off')
    
    plt.subplot(4,4,16)
    plt.imshow(np.array(filtered_image44),cmap='gray')
    plt.axis('off')    
    
    plt.savefig('parameter_sigma.png',dpi=512)
    plt.show()
    #####################################################################
    '''

    '''
    #3.不同去噪滤波算法的比较
    #####################################################################
    origin_image_path="./images/Cameraman.bmp"
    init_image_path="./images/Cameraman_noise.bmp"
    kernel_size=[7,7]
    sigma=[15,15]

    bif_image=bilateral_filter.bilateral_filtering_gray(init_image_path,kernel_size,sigma)
    gauss_image=Gaussian_filter.gaussian_filter(init_image_path,kernel_size,sigma[0])
    mean_image=mean_filter.mean_filter(init_image_path,kernel_size)
    median_image=median_filter.median_filter(init_image_path,kernel_size)

    origin_image_mat = np.array(Image.open(origin_image_path).convert('L'))
    init_image_mat = np.array(Image.open(init_image_path).convert('L'))

    plt.subplot(2,3,1)
    plt.title('original')
    plt.imshow(origin_image_mat,cmap="gray")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.title('Gaussian noisy')
    plt.imshow(init_image_mat,cmap="gray")
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.title('bilateral filtered')
    plt.imshow(np.array(bif_image),cmap="gray")
    plt.axis('off')

    plt.subplot(2,3,4)
    plt.title('Gaussian filtered')
    plt.imshow(np.array(gauss_image),cmap="gray")
    plt.axis('off')

    plt.subplot(2,3,5)
    plt.title('mean filtered')
    plt.imshow(np.array(mean_image),cmap="gray")
    plt.axis('off')

    plt.subplot(2,3,6)
    plt.title('median filtered')
    plt.imshow(np.array(median_image),cmap="gray")
    plt.axis('off')

    plt.savefig('filter_compare.png',dpi=512)
    plt.show()    
    #####################################################################
    '''

    '''
    #4.多粒度双边滤波算法
    #####################################################################
    origin_image_path="./images/lena512gray.bmp"
    init_image_path="./images/lena512gray_strong_noise.bmp"
    kernel_size=[11,11]
    multi_kernel_size=np.array([[11,11],[7,7],[3,3]])
    sigma=[20,20]
    bif=bilateral_filter.bilateral_filtering_gray(init_image_path,kernel_size,sigma)
    mul_bif_mat=multi_size_bilateral_filter.multi_size_bilateral_filtering(init_image_path,multi_kernel_size,sigma)

    origin_image_mat = np.array(Image.open(origin_image_path).convert('L'))
    init_image_mat = np.array(Image.open(init_image_path).convert('L'))

    plt.subplot(2,2,1)
    plt.title('original')
    plt.imshow(origin_image_mat,cmap="gray")
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.title('Gaussian noisy')
    plt.imshow(init_image_mat,cmap="gray")
    plt.axis('off')

    plt.subplot(2,2,3)
    plt.title('bilateral filtered')
    plt.imshow(np.array(bif),cmap="gray")
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.title('multi-size bilateral filtered')
    plt.imshow(mul_bif_mat,cmap="gray")
    plt.axis('off')
    plt.savefig('multi_size_bif',dpi=512)
    plt.show()
    #####################################################################
    '''
