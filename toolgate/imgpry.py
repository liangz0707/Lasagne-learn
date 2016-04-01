# coding:utf-8
__author__ = 'liangz14'
import logging
import numpy as np

from scipy.signal import convolve2d
from datetime import datetime

class LaplacianPyramid(object):
    """Represents the difference of gaussian pyramid for an image frame.

    Attributes:
       frame: An image frame.
       levels: The number of levels you'd like the pyramid to have.
    """

    def __init__(self, frame, levels):
        self._pyramid = []

        # Create the kernel outside the loop, so you only do this once.
        kernel = self.generating_kernel(0.4)

        for channel in range(len(frame[0,0])):
            current_pyramid = []
            reduction_pyramid = []
            expansion_pyramid = []
            channel_frame = frame[:,:,channel]

            reduction_pyramid.append(channel_frame)

            # Reduction
            for x in range(levels):
                out = convolve2d(channel_frame, kernel, 'same')
                channel_frame = out[::2,::2]
                reduction_pyramid.append(channel_frame)

            # Expansion
            for level in range(1, len(reduction_pyramid)):
                current_frame = reduction_pyramid[level]
                out = np.zeros((current_frame.shape[0]*2, current_frame.shape[1]*2))
                out[::2,::2] = current_frame
                # *4 because image becomes 4 times weaker with convolution
                expansion_pyramid.append(4*convolve2d(out, kernel, 'same'))

            # Remove last element.
            del reduction_pyramid[-1]

            # Subtract the pyramids.
            if (len(reduction_pyramid) != len(expansion_pyramid)):
                logging.critical("Pyramid size does not match.")
            else:
                for level in range(len(reduction_pyramid)):
                    reduction = reduction_pyramid[level]
                    expansion = expansion_pyramid[level]
                    laplacian = reduction - expansion[0:reduction.shape[0],
                                                      0:reduction.shape[1]]
                    current_pyramid.append(laplacian)

            # Append Laplacian Pyramid for the current channel.

            self._pyramid.append(current_pyramid)

    def generating_kernel(self, a):
        '''Returns a 5x5 generating kernel based on parameter a.
        '''
        w_1d = np.array([0.25 - a/2.0, 0.25, a, 0.25, 0.25 - a/2.0])
        return np.outer(w_1d, w_1d)

    @property
    def pyramid(self):
        """Return the Laplacian Pyramid.
        """
        return self._pyramid

import numpy as np
from skimage import io
from skimage import color
from skimage import data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def m1():
    img = data.lena()
    imgL = color.rgb2lab(img)
    c = LaplacianPyramid(imgL,3)
    pyr = c._pyramid
    plt.imshow(pyr[0][2])
    plt.show()

from skimage.transform import pyramid_laplacian

def m2():
    image = color.rgb2lab(data.astronaut())
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_laplacian(image, downscale=3))

    composite_image = np.zeros((rows, cols + cols / 2, 3), dtype=np.double)

    composite_image[:rows, :cols, :] = pyramid[0]

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows

    fig, ax = plt.subplots()
    ax.imshow(np.abs(composite_image[:,:,0]), cmap=cm.gray,interpolation="none")
    plt.show()
import math
import bicubic_2d
def m3():
    #下采样尺度
    scale = 3.0
    #只处理L通道
    image = color.rgb2lab(data.astronaut())[:,:,0]
    #计算金字塔深度
    pyr_depth = np.floor(math.log(np.min(image.shape),int(scale)))

    pyr_bicubic =[]
    pyr_hfreq = []
    c = image[:image.shape[0]-image.shape[0]%int(scale),:image.shape[1]-image.shape[1]%int(scale)]
    pyr_bicubic.append(c)
    for i in range(int(pyr_depth)-1):
        p = bicubic_2d.bicubic2d(pyr_bicubic[-1],1.0/scale)
        z = p[:p.shape[0]-p.shape[0]%int(scale),:p.shape[1]-p.shape[1]%int(scale)]
        pyr_bicubic.append(z)

        tmp = bicubic_2d.bicubic2d(p,scale)

        p = pyr_bicubic[-2] - tmp
        z = p[:p.shape[0]-p.shape[0]%int(scale),:p.shape[1]-p.shape[1]%int(scale)]
        pyr_hfreq.append(z)

    fig, ax = plt.subplots()
    ax.imshow(np.abs(pyr_hfreq[0]), cmap=cm.gray,interpolation="none")
    plt.show()
# m3()