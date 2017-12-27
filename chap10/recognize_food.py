import sys
import numpy as np
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'food_resnet-10_iter_10000.caffemodel'
DEPLOY_FILE = 'food_resnet_10_cvgj_deploy.prototxt'


#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

设置通道顺序和opencv一致(BGR)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

获取图片列表文件的路径，并生成保存结果
image_list = sys.argv[1]
result_list = '{}_results.txt'.format(image_list[:image_list.rfind('.')])

获取关键词
foods = open('/path/to/keywords.txt', 'rb').read().split()
with open(image_list, 'r') as f, open(result_list, 'w') as f_ret:
    for line in f.readlines():
        filepath, label = line.split()
        label = int(label)
        image = caffe.io.load_image(filepath)
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        各类对应概率在output中第一个元素中
        output = net.forward()
        probs = output['prob'][0]
        pred = np.argmax(probs)

        print('{}, predicted: {}, true: {}'.format(filepath, foods[pred], foods[label]))

        把结果保存到文件
        result_line = '{} {} {} {}\n'.format(filepath, label, pred, ' '.join([str(x) for x in probs]))
        f_ret.write(result_line)
