# coding:utf-8
__author__ = 'liangz14'
import numpy as np
from skimage import io
from skimage import color
from skimage import data
import matplotlib.pyplot as plt
import bicubic_2d
def rgb2ycbcr(img):
    """
    需要为255.0
    :param img:
    :return:
    """
    YCbCr = np.zeros(img.shape,dtype='float64')
    YCbCr[:,:,0] = 0.256789*img[:,:,0]+0.504129*img[:,:,1]+0.097906*img[:,:,2]+16.0
    YCbCr[:,:,1] = - 0.148233*img[:,:,0] - 0.290992*img[:,:,1]+ 0.439215*img[:,:,2]+128.0
    YCbCr[:,:,2] =   0.439215*img[:,:,0] - 0.367789*img[:,:,1] - 0.071426*img[:,:,2]+128.0
    return YCbCr

def ycbcr2rgb(YCbCr):
    img = np.zeros(YCbCr.shape,dtype='float64')
    img[:,:,0] = 1.164383*(YCbCr[:,:,0]-16) + 1.596027 * (YCbCr[:,:,2] - 128 )
    img[:,:,1] = 1.164383*(YCbCr[:,:,0]-16) - 0.391762 * (YCbCr[:,:,1] - 128) - 0.812969 *(YCbCr[:,:,2]-128)
    img[:,:,2] = 1.164383*(YCbCr[:,:,0]-16) + 2.017230 * (YCbCr[:,:,1] - 128)
    return img


def test_main():
    src_rgb_img = data.lena()

    src_YCrCb_img = rgb2ycbcr(src_rgb_img)
    p = ycbcr2rgb(src_YCrCb_img)

    c =np.asarray(p,dtype='uint8')

    plt.imshow(c)
    print src_rgb_img.dtype,src_rgb_img.shape,np.max(src_rgb_img),np.min(src_rgb_img)
    print c.dtype,c.shape,np.mean((c-src_rgb_img)**2),np.max(c),np.min(c)

    plt.show()

def test_main():
    src_rgb_img = data.lena()

    src_YCrCb_img = rgb2ycbcr(src_rgb_img)
    p = ycbcr2rgb(src_YCrCb_img)

    c =np.asarray(p,dtype='uint8')

    plt.imshow(c)
    print src_rgb_img.dtype,src_rgb_img.shape,np.max(src_rgb_img),np.min(src_rgb_img)
    print c.dtype,c.shape,np.mean((c-src_rgb_img)**2),np.max(c),np.min(c)

    plt.show()

def test_():
    src = data.lena()
    src_rgb_img = src[100:150,100:150,:]
    src_YCrCb_img = rgb2ycbcr(src_rgb_img)

    img_lab = np.zeros((src_YCrCb_img.shape[0]*3,src_YCrCb_img.shape[1]*3,3))
    img_lab[:,:,0] = bicubic_2d.bicubic2d(src_YCrCb_img[:,:,0],3.0)
    img_lab[:,:,1] = bicubic_2d.bicubic2d(src_YCrCb_img[:,:,1],3.0)
    img_lab[:,:,2] = bicubic_2d.bicubic2d(src_YCrCb_img[:,:,2],3.0)

    p = ycbcr2rgb(img_lab)

    c =np.asarray(p,dtype='uint8')

    plt.imshow(c)
    plt.show()
# test_()