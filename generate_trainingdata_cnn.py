# coding:utf-8
import numpy as np
import skimage.io as io
import skimage as simg
import skimage.color as clr
import skimage.filters as ft
import os
from scipy.misc import imresize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from toolgate import gabor
from skimage.morphology import disk
from toolgate.colormanage import rgb2ycbcr, ycbcr2rgb

__author__ = 'liangz14'


def get_train_set(img_lib):
    # 抽取字典~这个字典应该是HR-LR堆叠起来的结果
    scale = 3.0  # 放大倍数
    feat_scale = 3.0
    patch_size_l = 3  # 3
    patch_size_m = feat_scale * patch_size_l  # 9
    patch_size_h = scale * patch_size_l  # 9

    over_lap_l = 1  # 1
    over_lap_m = feat_scale * over_lap_l  # 3
    over_lap_h = scale * over_lap_l  # 3

    feature_lib = []
    target_lib = []
    raw_lib = []

    # 计算4个特征的卷积核,可以用4个方向的滤波作为特征
    f1 = np.asarray([[-1.0, 0, 0, 1.0]], dtype='float')
    f2 = np.asarray([[-1.0], [0], [0], [1.0]], dtype='float')
    f3 = np.asarray([[1.0, 0, 0, -2.0, 0, 0, 1.0]], dtype='float')
    f4 = np.asarray([[1.0], [0], [0], [-2.0], [0], [0], [1.0]], dtype='float')

    # 保存图像列表
    feature_img_list = []
    target_img_list = []  # 高ingpatch
    raw_img_list = []  # 原始patch

    # 每张图片计算八组特征
    for img in img_lib:
        s = img.shape
        # 要处理的原始图像
        image = img[0:s[0] - s[0] % 3, 0:s[1] - s[1] % 3, :]

        lim = imresize(image, 1.0/scale, interp='bicubic')
        # 缩小放大后的图像，用于提取训练特征
        mim = imresize(lim, feat_scale, interp='bicubic')
        # 缩小放大后的图像，用于提取需要计算的差值
        rim = imresize(lim, scale, interp='bicubic')

        # 提取y通道的方式进行计算 y[16 235] cbcr [16 240]
        image = np.asarray(rgb2ycbcr(image)[:, :, 0], dtype=float)
        mim = np.asarray(rgb2ycbcr(mim)[:, :, 0], dtype=float)
        rim = np.asarray(rgb2ycbcr(rim)[:, :, 0], dtype=float)

        patch = image - rim

        feature = np.zeros((8, mim.shape[0], mim.shape[1]))

        feature[0, :, :] = convolve2d(mim, f1, mode='same')
        feature[1, :, :] = convolve2d(mim, f2, mode='same')
        feature[2, :, :] = convolve2d(mim, f3, mode='same')
        feature[3, :, :] = convolve2d(mim, f4, mode='same')

        feature[4, :, :] = gabor.get_gabor(mim, 0.23, 0)
        feature[5, :, :] = gabor.get_gabor(mim, 0.23, 45)
        feature[6, :, :] = gabor.get_gabor(mim, 0.23, 90)
        feature[7, :, :] = gabor.get_gabor(mim, 0.23, 135)

        feature_img_list.append(feature)
        target_img_list.append(patch)
        raw_img_list.append(image)

    for i in zip(feature_img_list, target_img_list, raw_img_list):
        size_m = i[0].shape[1:]
        size_h = i[1].shape

        xgrid_m = np.ogrid[0:size_m[0]-patch_size_m: patch_size_m - over_lap_m]
        ygrid_m = np.ogrid[0:size_m[1]-patch_size_m: patch_size_m - over_lap_m]
        xgrid_h = np.ogrid[0:size_h[0]-patch_size_h: patch_size_h - over_lap_h]
        ygrid_h = np.ogrid[0:size_h[1]-patch_size_h: patch_size_h - over_lap_h]

        m = patch_size_m * patch_size_m * 4
        h = patch_size_h * patch_size_h

        for x_m, x_h in zip(xgrid_m, xgrid_h):
            for y_m, y_h in zip(ygrid_m, ygrid_h):
                target_lib.append(i[1][x_h:x_h+patch_size_h, y_h:y_h+patch_size_h])
                feature_lib.append(i[0][:, x_m:x_m+patch_size_m, y_m:y_m+patch_size_m])
                raw_lib.append(i[2][x_h:x_h+patch_size_h, y_h:y_h+patch_size_h])

    return target_lib, feature_lib, raw_lib


def read_img_train(cur_dir, down_time=20, scale=0.97):
    """
    读取目录下的图片:提取全部图片，并提取成单通道图像，并归一化数值[0,1]
    :param cur_dir:
    :return:
    """
    img_file_list = os.listdir(cur_dir)  # 读取目录下全部图片文件名
    img_lib = []
    for file_name in img_file_list:
        full_file_name = os.path.join(cur_dir, file_name)
        img_tmp = io.imread(full_file_name)  # 读取一张图片
        o_size = np.min(img_tmp)
        img_lib.append(img_tmp)
        # 多次下采样作为样本：
        for i in range(down_time):
            if np.min(img_tmp.shape) > o_size/20:
                img_tmp = imresize(img_tmp, size=scale, interp='bicubic')
                img_lib.append(img_tmp)
    return img_lib


def main_generate(ind, tr_num=500000):
    lib_path = os.getcwd()+'/../sr_image_data/train_data/%s' % 1
    res_path = './tmp_file/_%s_training_data.pickle' % ind
    res_raw_patch_path = './tmp_file/_%s_training_data_rawpatch.pickle' % ind

    img_lib = read_img_train(lib_path)
    print "图片数量为%d" % len(img_lib)
    # 得到训练的开始和结果
    target_lib, feature_lib, raw_lib = get_train_set(img_lib)

    if len(target_lib) > tr_num:
        s_factor = len(target_lib)/tr_num
        target_lib = target_lib[::s_factor]
        feature_lib = feature_lib[::s_factor]
        raw_lib = raw_lib[::s_factor]

    training_data = (target_lib, feature_lib)

    import cPickle

    with open(res_raw_patch_path, 'wb') as f:
        cPickle.dump(raw_lib, f, 1)

    with open(res_path, 'wb') as f:
        cPickle.dump(training_data, f, 1)

    print target_lib[1].shape
    print feature_lib[1].shape
    print len(target_lib)
    print len(feature_lib)

if __name__ == '__main__':
    # 修改这个结果就可以生成不同文件夹中的训练数据
    main_generate("cnn_1")
