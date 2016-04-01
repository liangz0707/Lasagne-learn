# coding:utf-8
import skimage.data as data
import skimage.color as color
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d
import numpy as np
__author__ = 'liangz14'


def get_gabor(img, frequency, theta):
    kernel = np.real(gabor_kernel(frequency, theta=theta*np.pi/180.0))
    result = convolve2d(img, kernel, mode='same')
    return result


def main():
    # theta 是顺时针旋转角度
    # frequency 是频率
    kernel = np.real(gabor_kernel(0.1, theta=30*np.pi/180.0))

    img = data.lena()
    img = color.rgb2gray(img)

    img = convolve2d(img, kernel, mode='same')
    print img
    axe2 = plt.axes([0, 0, 1, 1])
    axe2.imshow(img, cmap=cm.gray)
    axe = plt.axes([0, 0, 0.2, 0.2])
    axe.imshow(kernel, cmap=cm.gray)

    plt.show()

if __name__ == "__main__":
    main()
