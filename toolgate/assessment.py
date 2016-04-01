# coding:utf-8
__author__ = 'liangz14'
import numpy as np
import math

def psnr(img1, img2):
    """
    psnr计算过程数据范围需要在 0 - 255 之间，并且需要是二维矩阵
    :param img1:
    :param img2:
    :return:
    """
    mse = np.sum((img1/255.0 - img2/255.0) ** 2)
    # print img1
    if mse == 0:
        return 100
    N = img1.shape[1] * img1.shape[0]
    return 10 * math.log10(N / mse)


def mse(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    return mse


def rmse(img1,img2):
    rmse = np.sqrt(np.mean( (img1 - img2) ** 2 ))
    return rmse