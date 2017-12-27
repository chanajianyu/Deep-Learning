## Metric Learning度量学习 with Siamese Network
### step 1      准备MNIST数据


按照第八章准备MNIST数据的方法生成图片，然后执行
> ln -s /path/to/mnist mnist

在目录下链接mnist图片所在的目录

思路:
    1. 利用之前用过的lmdb实现Siamese训练
    2. 生成两个lmdb
           其中数据顺序一一对应
           相同数字，则标签为1，否则标签为0
    3. 对训练时用到的prototxt做修改
            用两个data layer分别读取两个数据
            经过CNN之后计算转换后特征距离      
           
    
    
    
    
    
    
    
    
    
    
    
    
### step 2
> python gen_pairwise_imglist.py

生成成对图片的列表

### step 3
> /path/to/caffe/build/tools/convert_imageset ./ train.txt train_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ train_p.txt train_p_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val.txt val_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val_p.txt val_p_lmdb --gray  

执行脚本，得到train.txt、train_p.txt、val.txt、val_p.txt
生成lmdb

### step 4

完整solver和网络结构定义的训练过程
> /path/to/caffe/build/tools/caffe train -solver mnist_siamese_solver.prototxt -log_dir ./ 

训练模型

### step 5

> python visualize_result.py

进行结果可视化

## 预训练模型下载链接
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/mnist_siamese_iter_20000.caffemodel
或  
http://pan.baidu.com/s/1qYk5MDQ