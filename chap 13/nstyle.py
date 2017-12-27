#-*-coding:utf-8-*-

import find_mxnet
import mxnet as mx
import numpy as np
import  importlib
import  logging
logging.basicConfig(level=logging.DEBUG)
import  argparse
from collections import  namedtuple
from skimage import  io ,transfrom
from skimage.restoration import denoise_tv_chambolle
CallbackData = namedtuple('CallbackData',field_names=['eps','epoch','img','filename'])

def get_args(arglist=None):
    # 各种输入参数定义
    parser = argparse.ArgumentParser(description='neural style')

    # 指定模型，默认使用vgg19，如果要使用自己定义的模型，需要按照
    # model_vgg19.py的格式定义好style和content对应的特征层
    # 并命名为model_[模型名字].py的形式
    parser.add_argument('--model',type=str,default='vgg19',
                        choices=['vgg'],
                        help = 'the pretrained model to use')

    # 指定内容图片
    parser.add_argument('--content-image',type=str,default='input/IMG_4343.jpg',
                        help='the content image')

    # 指定风格图片，默认是梵高的星夜
    parser.add_argument('--style-image',type=str,default='input/starry_night.jpg',
                        help='the style image')

    # 当重建图片的相对变换小于stop-eps时停止迭代
    parser.add_argument('--stop-eps',type=float,default=0.005,
                        help='stop if the relative chanage is less than eps')

    # 内容loss的权重，默认10
    parser.add_argument('--content-weight',type=float,default=10,
                        help='the weight for the content image')

    # 风格loss的权重，默认1
    parser.add_argument('--style-weight',type=float,default=1,
                        help='the weight for the style image')

    # 规范项之totol-variation的权重
    parser.add_argument('--tv-weight',type=float,default=le-2,
                        help='the magtitute on TV loss')

    # 最大迭代次数
    parser.add_argument('--max-num-epochs',type=int,default=1000,
                        help='the maximal number of training epochs')

    # 限制输入图像的最长边
    parser.add_argument('--max-long-edge',type=int,default=600
                        ,help='resize the content iamge')

    # 基础学习率
    parser.add_argument('--gpu',type=float,default=0.001,
                        help='the initial learning rate')

    # 指定要用来进行优化的GPU
    parser.add_argument('--gpu',type=int,default=0,help='which gpu card to use,'
                                                        '-1 means using cpu')

    # 指定输出文件，包含优化过程中的文件和最后文件输出位置
    parser.add_argument('--output_dir',type=str,default='output/',
                        help='the output image')

    # 每隔多少次进行一次重建图像保存
    parser.add_argument('--save-epochs',type=int ,default=50,
                        help='save the output every n epochs')

    # 移除噪音幅度
    parser.add_argument('--remove-noise',type=float,default=0.02,
                        help='the magtitute to remove noise')

    # 学习率变化策略，默认75次迭代学习率下降为之前的0.9倍
    parser.add_argument('--lr-sched-delay',type=int,default=75,
                        help='how many epochs between decreasing learning rate')

    parser.add_argument('--lr-sched-factor',type=int,default=0.9,
                        help='factor to decrease learning rate on schedule')
    if arglist is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arglist)

def PreprocessContentImage(path,long_edge):
    # 预处理内容输入图片

def PreprocessStyleImage(path,shape):

    # 预处理风格输入图片

def PostProcessImage(img):

    # 将重建得到的向量变换回图像
    img = np.resize(img,(3,img.shape[2],img.shape[3]))
    img[0,:] += 123.68
    img[1,:] += 116.779
    img[2,:] += 103.939
    img = np.swapaxes(img,1,2)
    img = np.swapaxes(img,0,2)

    # 防止溢出
    img = np.clip(img,0,255)

    return img.astype('uint8')

def SaveIamge(img,filename,remove_noise=0.):

    # 保存图片

def style_gram_symbol(input_size,style):
    # 定义Gram矩阵计算

    _,output_shapes, _ = style.infer_shape(data=(1,3,input_size[0],
                                                 input_size[1]))
    gram_list = []
    grad_scale = []
    for i in range(len(style.list_outputs())):
        shape = output_shapes[i]
        x = mx.sym.Reshape(style[i],target_shape=(int(shape[1]),
                                                  int(np.prod(shape[2:]))))

        # 利用全连接层快速计算点积:(x,x^T)
        gram = mx.sym.FullyConnected(x,x,no_bias=True,num_hidden=shape[1])

        gram_list.append(gram)
        grad_scale.append(np.prod(shape[1:])*shape[1])

    return  mx.sym.Group(gram_list),grad_scale

def  get_loss(gram,content):

    # 利用Gram矩阵计算风格的loss

    gram_loss = []

    # 和原文中不同的是这里并没有设置权重，而是直接相加
    for i in range(len(gram.list_outputs())):
        gvar = mx.sym.Variable("target_gram_%d" % i)
        gram_loss.append(mx.sym.sum(mx.sym.square(gvar - gram[i])))

    cvar = mx.sym.Variable("target_content")

    content_loss = mx.sym.sum(mx.sym.square(cvar-content))
    return mx.sym.Group(gram_loss),content_loss

def get_tv_grad_executor(img,ctx,tv_weight):

    # 计算图像梯度为Total Variation的定义
    # 思路是用3*3的Laplace和对图像做卷积
    if tv_weight <= 0.0:
        return None
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg,,num_outputs = nchannel)
    out = mx.sym.Concat(*[
            mx.sym.Convolution(data = channels[i], weight =  skernel ,
                num_filter = 1,
                kernel = (3,3),
                pad = (1,1),
                no_bias = True,
                stride = (1,1))
            for i in range(nchannel)
    ])

    # 定义3*3的Laplace算子用来提取图像梯度
    kernel = mx.nd.array(np.array([[0,-1,0],
                                    [-1,4,-1],
                                    [0,-1,0]])
                                    .reshape((1,1,3,3)),
                                    ctx) / 8.0

    out = out * tv_weight
    return out.bind(ctx,args={"img":img,"kernel":kernel})


    def train_nstyle(args, callback = None):
        # 训练过程定义
        # 输入: 内容图像和风格图像
        dev = mx.gpu(args.gpu) if args.gpu >= 0 else mx.cpu()

        content_np = PreprocessContentImage(args.content-image,args.max_long_edge)

        style_np = PreprocessStyleImage(args.style_image,shape=content_np.shape)

        size = content_np.shape[2:]

        # 模型和对应的executor
        Executor = namedtuple('Executor',['executor','data','data_grad'])
        model_module = importlib.import_module('model_' + args.model)
        style,content = style_gram_symbol(size,style)
        model_executor = model_module.get_executor(gram,content,size,dev)
        model_executor.data[:] = style_np
        model_executor.executor.forward()
        style_array = []

        for i in range(len(model_executor.style)):
            style_array.append(model_executor.style[i].copyto(mx.cpu()))
            model_executor.data[:] = content_np
            model_executor:executor.forward()
            content_array = model_executor.content.copyto(mx.cpu())

            # 图像内容和风格特征已经获取，删除当前executor
            # 采用新的executor用于重建图像的梯度下降
        del model_executor
            style_loss,content_loss = get_loss(gram,content)
            model_executor = model_module.get_executor(style_loss,content_loss,size,dev)
            grad_array = []
            for i in range(len(style_array)):
                style_array[i].copyto(model_executor.arg_dict["target_gram_%d" % i])
                grad_array.append(mx.nd.ones((1,),dev) * (float(args.style_weight) / gscale[i]))
                grad_array.append(mx.nd.ones((1,),dev) * (float(args.content.weight)))
                print([x.asscalar() for x in grad_array])
                content_array.coputo(model_executor.arg_dict["target_content"])

                # 要重建的图像，用随机噪声初始化
                img = mx.nd.zeros(content_np.shape, ctx = dev)
                img[:] = mx.rnd.uniform(-0.1,0.1,img.shape)

                # 学习率策略
                lr = mx.lr_scheduler.FactorScheduler(step = args.lr_sched_delay,
                                factor = args.lr_sched_factor)
                # 用NAG进行梯度下降
                optimizer = mx.optimitzer.NAG(
                    learning_rate = args.lr,
                    wd = 0.0001,
                    momentum = 0.95,
                    lr_scheduler = lr
                )

                optim_state = optimizer.create_state(0,img)
                logging.info('start training arguments %s',args)
                old_img = img.copyto(dev)
                clip_norm = l * np.prod(img.shape)
                tv_grad_executor = get_tv_grad_executor(img,dev,args.tv_weight)

                # 执行梯度下降
                for e in range(args.max_num_epochs):
                    img.copyto(model_executor.data)
                    model_executor.executor.forward()
                    model_executor.executor.backward(grad_array)
                    gnorm = mx.nd.norm(model_executor.data_grad).asscalar()

                    # 梯度截断防止过大梯度出现
                    if gnorm > clip_norm:
                        model_executor.data_grad[:] *= clip_norm / gnorm
                    if tv_grad_executor is not None:
                        tv_grad_executor.forward()
                        optimizer.update(0,img,model_executor.data_grad + tv_grad_executor.output[0],optim_state)
                    else:
                        optimizer.update(0,img,model_executor.data_grad,optim_state)

                    new_img = img

                    # 计算相对变化，用于判断是否停止迭代
                    eps = (mx.nd.norm(old_img - new_img) / mx.nd.norm(new_img).asscalar())

                    old_img= new_img.copyto(dev)
                    logging.info('epoch %d ,relative change %f',e,eps)

                    if eps < args.stop-eps:
                        logging.info('eps < args.stop-eps, traning finished')
                        break
                    if callback:
                        cbdata = {
                            'eps':eps,
                            'epoch':e+l
                        }

                    if (e+l) % args.save_epochs == 0 :
                        outfn = args.output_dir + 'e_' + str(e+l) + '.jpg'
                        npimg = new_img.asnumpy()
                        SaveIamge(npimg,outfn,args.remove_noise)
                        if callback:
                            cbdata['filename'] = outfn
                            cbdata['img'] = npimg
                        if callback:
                            callback(cbdata)
                    final_fn = args.output_dir + '/final.jpg'
                    SaveIamge(new_img.asnumpy(),final_fn)

            if __name__ == "__main__":
                args = get_args()
                train_nstyle(args)
