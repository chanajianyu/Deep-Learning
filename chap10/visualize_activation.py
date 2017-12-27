import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('/path/to/caffe/python')
import caffe

WEIGHTS_FILE = 'food_resnet-10_iter_10000.caffemodel'
DEPLOY_FILE = 'food_resnet_10_cvgj_deploy.prototxt'

指定最后一层响应图的blob名称
FEATURE_MAPS = 'layer_512_1_sum'

指定最后一层全连接层的名称
FC_LAYER = 'fc_food'

#caffe.set_mode_cpu()
net = caffe.Net(DEPLOY_FILE, WEIGHTS_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_raw_scale('data', 255)

预处理数据，从RGB到BGR
transformer.set_channel_swap('data', (2, 1, 0))

image_list = sys.argv[1]

采用jet作为响应图的热度图方案
cmap = plt.get_cmap('jet')
with open(image_list, 'r') as f:
    for line in f.readlines():
        filepath = line.split()[0]
        image = caffe.io.load_image(filepath)

        取消下面两行注解，用原大小的图生成分辨率更高的响应图
        # uncomment the following 2 lines to forward with
        # original image size and corresponding activation maps
        #transformer.inputs['data'] = (1, 3, image.shape[0], image.shape[1])
        #net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        pred = np.argmax(output['prob'][0])

        获得最后一层的7*7响应图，一共512通道
        feature_maps = net.blobs[FEATURE_MAPS].data[0]

        获取最后一层全连接的参数值
        fc_params = net.params[FC_LAYER]
        获取预测类别对应的参数值
        fc_w = fc_params[0].data[pred]
        #fc_b = fc_params[1].data[pred]

        计算该类别对应的响应图
        activation_map = np.zeros_like(feature_maps[0])
        for feature_map, w in zip(feature_maps, fc_w):
            activation_map += feature_map * w
        #activation_map += fc_b

        可视化:左图为原图，中间为响应图，右图为热度图和原图的叠加
        原图重新表示为opencv可接受的格式
        # Visualize as
        # left: original image
        # middle: activation map
        # right: original image overlaid with activation map in 'jet' colormap
        image = np.round(image*255).astype(np.uint8)
        h, w = image.shape[:2]

        响应图放大到和原图一样大，为了可视化效果用CUBIC插值让变化平滑
        activation_map = cv2.resize(activation_map, (w, h), interpolation=cv2.INTER_CUBIC)

        归一化，然后重新表示为opencv可接受的格式
        activation_map -= activation_map.min()
        activation_map /= activation_map.max()

        生成热度图
        activation_color_map = np.round(cmap(activation_map)[:, :, :3]*255).astype(np.uint8)

        生成三通道响应图方便拼接
        activation_map = np.stack(np.round([activation_map*255]*3).astype(np.uint8))
        activation_map = activation_map.transpose(1, 2, 0)

        生成叠加热度图
        overlay_img = image/2 + activation_color_map/2

        横向拼接
        vis_img = np.hstack([image, activation_map, overlay_img])

        通道交换:RGB--->BGR
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

        cv2.imshow('Activation Map Visualization', vis_img)
        cv2.waitKey()

       默认图像缩放到224*224,最后得到响应图大小是7*7,

        在食物自身特征最明显的区域出现高的响应

        卷积有个性质是同变性，原图中物体移动到那里，响应图中的响应就移动到哪里。


























