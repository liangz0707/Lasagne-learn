# coding:utf-8
__author__ = 'liangz14'

# coding:utf-8
import random
import math
import numpy as np
import scipy.io
import skimage.io
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import bicubic_2d
from scipy.misc  import imresize
from  scipy.signal import convolve2d

def onto_unit(x):
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)
def visualize_patches(B):
    # assume square
    mpatch = int(math.floor(math.sqrt(B.shape[0])))
    npatch = mpatch

    m = int(math.floor(math.sqrt(B.shape[1])))
    n = int(math.ceil(B.shape[1] * 1.0 / m))
    collage = np.zeros((m*mpatch, n*npatch))
    for i in xrange(m):
        for j in xrange(n):
            try:
                patch = B[:, i*n + j]
            except IndexError:
                continue
            patch = onto_unit(patch.reshape((mpatch, npatch)))
            collage[i*mpatch:(i+1)*mpatch, j*npatch:(j+1)*npatch] = patch


def getTrainSet(img_lib,patch_dic_num):
    img_num = len(img_lib) #图像个数
    #抽取字典~这个字典应该是HR-LRduo堆叠起来的结果
    scale = 3.0 #放大倍数
    feat_scale=2.0
    patch_sizel = 3 #高清patch的尺寸， 对应底分辨率的patch就是   patch_size/scale
    patch_sizeh = patch_sizel*scale
    patch_sizem = patch_sizel*feat_scale

    HRpatch_lib = []
    Fpatch_1_lab = []
    Fpatch_2_lab = []
    Fpatch_3_lab = []
    Fpatch_4_lab = []

    F_1_lab = []
    F_2_lab = []
    F_3_lab = []
    F_4_lab = []
    H_lib = []

    f1=np.asarray([[-1,0,1]],dtype='float64')
    f2=np.asarray([[-1],[0],[1]],dtype='float64')
    f3=np.asarray([[1,0,-2,0,1]],dtype='float64')
    f4=np.asarray([[1],[0],[-2],[0],[1]],dtype='float64')
    #准备高清图像的高频部分，和LR的四种特征
    for i in range(img_num):
        tmp = bicubic_2d.bicubic2d(img_lib[i],1/3.0)
        tmp2 = bicubic_2d.bicubic2d(tmp,2.0)
        tmp3 = bicubic_2d.bicubic2d(tmp,3.0)
        F_1_lab.append(convolve2d(tmp2,f1,mode='same'))
        F_2_lab.append(convolve2d(tmp2,f2,mode='same'))
        F_3_lab.append(convolve2d(tmp2,f3,mode='same'))
        F_4_lab.append(convolve2d(tmp2,f4,mode='same'))
        Hm = img_lib[i]
        Hm =Hm[ :Hm.shape[0]-Hm.shape[0]%3,:Hm.shape[1]-Hm.shape[1]%3]
        H_lib.append(Hm - tmp3)

    for i in range(patch_dic_num):
        img_i =random.randint(0,img_num-1) #得到去那一张图片。
        img_temp = img_lib[img_i]
        y,x = [random.randint(0,img_temp.shape[d]/3 - 4) for d in (0,1)]
        ym = y * 2
        xm = x * 2
        yh = y * 3
        xh = x * 3

        HRpatch = H_lib[img_i][yh:yh+patch_sizeh,xh:xh+patch_sizeh]
        F1 = F_1_lab[img_i][ym:ym+patch_sizem,xm:xm+patch_sizem]
        F2 = F_2_lab[img_i][ym:ym+patch_sizem,xm:xm+patch_sizem]
        F3 = F_3_lab[img_i][ym:ym+patch_sizem,xm:xm+patch_sizem]
        F4 = F_4_lab[img_i][ym:ym+patch_sizem,xm:xm+patch_sizem]

        #高清patch
        HRpatch_lib.append(HRpatch)
        Fpatch_1_lab.append(F1)
        Fpatch_2_lab.append(F2)
        Fpatch_3_lab.append(F3)
        Fpatch_4_lab.append(F4)


    ylist = []
    xlist = []
    for i in range(len(HRpatch_lib)):
        H=HRpatch_lib[i]
        F=np.zeros((patch_sizem,patch_sizem,4))
        F[:,:,0]= Fpatch_1_lab[i]
        F[:,:,1]= Fpatch_2_lab[i]
        F[:,:,2]= Fpatch_3_lab[i]
        F[:,:,3]= Fpatch_4_lab[i]

        xx=[]

        normalization_m = math.sqrt(np.sum(F**2))
        if normalization_m > 1:
            xx = F/normalization_m
        else:
            xx = F

        yy=H-np.mean(H)

        ylist.append(yy.reshape((patch_sizeh*patch_sizeh)))
        xlist.append(xx.reshape((patch_sizem*patch_sizem*4)))
    return (xlist,ylist)

#读取目录下的图片:提取全部图片，并提取成单通道图像，并归一化数值[0,1]
def readImgTrain(cur_dir):
    img_file_list =os.listdir(cur_dir) #读取目录下全部图片文件名
    img_lib = []
    for file_name in img_file_list:
        full_file_name = os.path.join(cur_dir,file_name)
        img_tmp = skimage.io.imread(full_file_name) #读取一张图片
        if img_tmp.ndim !=2:
            img_tmp =  skimage.color.rgb2lab(img_tmp)
            img_tmp = img_tmp[:,:,0]
            img_tmp = img_tmp /100
        else:
            img_tmp = skimage.img_as_float(img_tmp)
        #print img_tmp.dtype , img_tmp.shape , np.max(img_tmp),np.min(img_tmp)
        img_lib.append(img_tmp)
    return img_lib

#获取目录返回patch对
def getHLPair(path,k=3):
    #得到img_lib：得到全部的图片
    img_lib = readImgTrain(path)
    #得到patch的对
    return getTrainSet(img_lib,10000)

# (xlist,ylist) = getHLPair('./data')