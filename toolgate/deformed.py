# -*- coding: utf-8 -*-
import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv  # 接触theano进行卷积操作
import time
__author__ = 'Zhe'

class deformed_patch(object):
    def __init__(self):
        pass

    def getXYGrad(self, input):
        ylen = input.shape[0]
        xlen = input.shape[1]
        #这个输入需要改变成一维的，为了借助Theano的函数
        src = T.dmatrix() #定义输入图像数组 二维、单通道

        #这个滤波器
        filter_shape = np.asarray([2,1,2,2],dtype='int')
        image_shape = np.asarray([1,1,ylen,xlen],dtype='int') #就是实际图像的大小

        #从src向dst变化
        src = src.reshape((1, 1, ylen, xlen))

        #计算两个方向的梯度
        Wx=np.asarray([[1, -1], [0, 0]], dtype='float64') #结果是W+Wx
        Wy=np.asarray([[1, 0], [-1, 0]], dtype='float64') #
        W=np.zeros(filter_shape)
        W[0,0]=Wx
        W[1,0]=Wy

        src_xd = conv.conv2d(
                input=src,
                filters=W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )#卷积会忽略边界

        getGradXY = theano.function([src],src_xd)

        #input = [[1,1,1,1,1,1,1],[2,2,2,2,2,2,2],[3,3,3,3,3,3,3],[4,4,4,4,4,4,4],[5,5,5,5,5,5,5],[6,6,6,6,6,6,6],[7,7,7,7,7,7,7]]
        input = np.asarray(input,dtype='float64')
        input=input.reshape((1,1,ylen,xlen))
        #print input.shape
        c = getGradXY(input)
        #print c.shape
        #这里得到的c[0]是x方向的导数，c[1]是y方向的导数
        c = c.reshape((c.shape[1],c.shape[2],c.shape[3]))
        #plt.imshow(c[1])
        return (c[0],c[1])

    #输入两块相同大小并且有略微变形的patch，计算变形程度(U,V)
    def getUV(self, img_in, img_out):
        xlen = img_in.shape[1]
        ylen = img_in.shape[0]

        img_out # = img_out.reshape((1, 1, ylen, xlen))
        img_in #= img_in.reshape((1, 1, ylen, xlen))

        # 定义输入和目标
        src_c = T.dmatrix()
        dst_c = T.dmatrix()

        # 对于每一个输入都减去均值
        src_in = src_c - T.sum(src_c)/(xlen*ylen)
        dst_in = dst_c - T.sum(dst_c)/(xlen*ylen)

        # 为了进行卷积，需要重新修改shape
        src = src_in.reshape((1, 1, ylen, xlen))
        dst = dst_in.reshape((1, 1, ylen, xlen))

        # 声明两个方向平移域 u和v
        u = theano.shared(
                value=np.zeros([1, 1, ylen, xlen], dtype=theano.config.floatX),
                borrow=True
            )
        v = theano.shared(
                value=np.zeros([1, 1, ylen, xlen], dtype=theano.config.floatX),
                borrow=True
            )
        print u"声明结束"

        filter_shape = np.asarray([2,1,2,2],dtype='int')
        image_shape = np.asarray([1,1,ylen,xlen],dtype='int') #就是实际图像的大小

        #计算两个方向的梯度
        Wx=np.asarray([[-1,1],[0,0]],dtype='float64') #结果是W+Wx
        Wy=np.asarray([[-1,0],[1,0]],dtype='float64') #
        W=np.zeros(filter_shape)
        W[0,0]=Wx
        W[1,0]=Wy
        print W.shape

        #首先需要计算梯度
        src_grad = conv.conv2d(
                input=src,
                filters=W,
                filter_shape=filter_shape,
                image_shape=image_shape,
                border_mode="full"
            )#卷积会忽略边界

        print u"梯度计算结束"

        ''' '''
        gu = conv.conv2d(
                input=src,
                filters=W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )#卷积会忽略边界
        gv = conv.conv2d(
                input=src,
                filters=W,
                filter_shape=filter_shape,
                image_shape=image_shape
            )#卷积会忽略边界

        cost = T.sum((dst - src+u*src_grad[0,1,0:ylen,0:xlen]+v*src_grad[0,0,0:ylen,0:xlen])**2)+0.1*T.mean(gv**2+gu**2)+0.15*T.mean(v**2+u**2)
        params = [u, v]
        grads = T.grad(cost, params)#所有的梯度
        updates = [
            (param_i, param_i - 0.15 * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        print u"create"
        train_model = theano.function(
            [src_c, dst_c],
            [cost, src_grad],
            updates=updates,
        )

        #theano.printing.pydotprint(train_model, outfile="tree.png", var_with_name_simple=True)
        print u"compute"
        g=[]
        a = time.time()

        for i in xrange(10):
            [c,g] = train_model(img_in,img_out)
            print c
        b = time.time()
        #print b-a

        '''
        p = theano.function([src_in],src_grad)
        c = p(img_in)

        print c.dtype,c.shape
        '''
        img_in=img_in.reshape((1, 1, ylen, xlen))
        img_in=img_in[0,0,0:ylen,0:xlen]
        cp=img_in-u.get_value()*g[0,1,0:ylen,0:xlen]-v.get_value()*g[0,0,0:ylen,0:xlen]
        return (u.get_value(),v.get_value(),cp)

    def deform(self, img_in, img_out, grad_x, grad_y):
        ylen= img_in.shape[1]
        xlen  = img_in.shape[0]

        # 定义输入和目标
        src = T.dmatrix()
        dst = T.dmatrix()

        # 声明两个方向平移域 u和v
        gu = theano.shared(value=grad_x, borrow=True)
        gv = theano.shared(value=grad_y, borrow=True)

        u = theano.shared(
                value=np.zeros([ylen, xlen], dtype=theano.config.floatX),
                borrow=True
            )
        v = theano.shared(
                value=np.zeros([ylen, xlen], dtype=theano.config.floatX),
                borrow=True
            )
        cost = T.mean((src+u*gu+v*gv-dst)**2)#+0.005*T.mean(v**2+u**2) #+0.1*T.mean(gv**2+gu**2)

        params = [u, v]
        grads = T.grad(cost, params)#所有的梯度
        updates = [
            (param_i, param_i - 0.5 * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        train_model = theano.function(
            [src, dst],
            cost,
            updates=updates,
        )
        c=None
        for i in xrange(5000):
            c = train_model(img_in, img_out)
        cp = img_in+u.get_value()*gu.get_value()+v.get_value()*gv.get_value()
        return cp,c
