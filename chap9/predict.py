import sys
import numpy as np
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'freq_regression_iter_10000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'
MEAN_VALUE = 128

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

Transformer是Caffe中用于对图片进行预处理并转换成分通道形式的模块
用deploy.prototxt中的shape作为图像的shape

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

转换 高度*宽度*通道 -----> 通道*高度*宽度
transformer.set_transpose('data', (2,0,1))

减去均值
transformer.set_mean('data', np.array([MEAN_VALUE]))
缩放系数
transformer.set_raw_scale('data', 255)

image_list = sys.argv[1]

获取batch_size
batch_size = net.blobs['data'].data.shape[0]
with open(image_list, 'r') as f:
    i = 0
    filenames = []
    for line in f.readlines():
        filename = line[:-1]
        filenames.append(filename)

        读取一张图片，默认读取图片为0-1之间的浮点数矩阵
        image = caffe.io.load_image(filename, False)

        按照transformer的设置转化为网络接受的格式并进行减均值等预处理
        transformed_image = transformer.preprocess('data', image)

        按对应位置装载到data的blob里
        net.blobs['data'].data[i, ...] = transformed_image
        i += 1

        装载数据够一个batch后，执行一次向前计算
        if i == batch_size:
            output = net.forward()
            freqs = output['pred']

            从结果中读取该batch所有结果并显示
            for filename, (fx, fy) in zip(filenames, freqs):
                print('Predicted frequencies for {} is {:.2f} and {:.2f}'.format(filename, fx, fy))

            重新开始给batch计数
            i = 0
            filenames = []


处理图像:
    使用caffe的IO模块(背后实现是scikit-image)读取图片，
    用Caffe的Transformer模块对图片进行减均值、通道变换预处理
