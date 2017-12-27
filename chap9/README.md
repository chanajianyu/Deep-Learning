## Step 1

生成·随机噪声样本
> python make_noises.py

## Step 2
产生3个数据集列表
> python gen_label.py

## Step 3

基于txt文件中列表，产生对应HDF5文件
这是执行命令顺序
> python gen_hdf5.py train.txt  
> python gen_hdf5.py val.txt

## Step 4
训练网络
> /path/to/caffe/build/tools/caffe train -solver solver.prototxt

## Step 5

> python predict.py test.txt

## Visualize Conv1 Kernels
> python visualize_conv1_kernels.py


第九章笔记：

回归场景是让模型预测的实数值和标签值尽量一致

9.1 预测值和标签值的欧式距离
      用于分类卷积神经网络结构，前面部分是若干各种方式连接的卷积层，最后全连接层，得到向量，经过Softmax层输出预测的分类概率。
      就是不断把一个向量变换成另一个向量

9.2 制作多标签HDF5数据
      多标签

      9.2.3 网络结构和Solver定义

            网络结构是试着在LeNet-5加一层，然后改动卷积核大小和stride，让最后进入全连接前的维度不会太高
      9.2.4 训练网络
            问题:
                回归比分类更难训练，用Caffe时容易出现不收敛问题，或者收敛到很大的loss上。
            原因:
                1. 参数设置不合适，可以尝试减小参数初始化时浮动大小，减小学习率或改变梯度下降方法
                2. 查看是否使用cuDNN，
                      通常表现是GPU不训练不收敛，
                      换成CPU或者删除cuDNN就可以

      9.2.5 批量装载图片并利用GPU预测
            有GPU时，可以把多张图片同时装载到显存中，然后运行一次得到多张图片结果，实现并行。
            首先:
                改写train_val.prototxt生成一个专门部署网络结构描述文件deploy.prototxt

      9.2.6 卷积核可视化
            
