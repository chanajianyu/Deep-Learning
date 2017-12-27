import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('/path/to/caffe/python')
import caffe

卷积核大小为5*5，放大方便观察
ZOOM_IN_SIZE = 50

每个卷积核放大后之间的间隔
PAD_SIZE = 4


WEIGHTS_FILE = 'freq_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'

net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

通过param属性取得指定层的参数的值
kernels = net.params['conv1'][0].data

归一化到0-1之间方便可视化
kernels -= kernels.min()
kernels /= kernels.max()

进行缩放，有5*5----->50*50
zoomed_in_kernels = []
for kernel in kernels:
    zoomed_in_kernels.append(cv2.resize(kernel[0], (ZOOM_IN_SIZE, ZOOM_IN_SIZE), interpolation=cv2.INTER_NEAREST))

# plot 12*8 squares kernels
half_pad = PAD_SIZE / 2
padded_size = ZOOM_IN_SIZE+PAD_SIZE

利用numpy的pad方法直接在卷积核图像的两边补上half_pad这么多像素的白边
padding = ((0, 0), (half_pad, half_pad), (half_pad, half_pad))

padded_kernels = np.pad(zoomed_in_kernels, padding, 'constant', constant_values=1)

一共96个卷积核，画成8*12的排列
padded_kernels = padded_kernels.reshape(8, 12, padded_size, padded_size).transpose(0, 2, 1, 3)
kernels_img = padded_kernels.reshape((8*padded_size, 12*padded_size))[half_pad:-half_pad, half_pad: -half_pad]

可视化生成的图像，包含所有第一层96个卷积核
plt.imshow(kernels_img, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.show()


可视化结果，得到第一层卷积核成功学到各种不同频率成分，例子中判断频率高低的关键。
高频成分通过高频的卷积核响应向下一层传递，
低频则通过变化缓慢的卷积核响应传向下一层
