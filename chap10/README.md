## 第十章 迁移学习和模型微调

从搜索引擎上下载关键词对应的图片，并利用模型微调进行训练，以及最后的模型评估，分析和可视化。

具体用法参考本书第十章内容。

书中例子对应的预训练模型可以在下面地址下载：  
https://github.com/frombeijingwithlove/dlcv_book_pretrained_caffe_models/blob/master/food_resnet-10_iter_10000.caffemodel  
或  
http://pan.baidu.com/s/1jHRLsLw


10.1.1 通过关键词和图片搜索引擎下载图片
    思路:
      通过在搜索引擎中搜索给定关键词
      对返回结果的静态网页版本的源代码进行匹配找到所有图片URL，下载
    实现思路:
      在一个文本文件中以UTF-8格式保存要下载食品关键词列表
      编写脚本自动为每个关键词分配多进程
      进行网页源码获取和图片下载
    总结:
      先在搜索引擎中输入关键词
      返回结果后观察浏览器地址栏是否有规律
          比如关键词是哪个字段，翻页后变化的是哪个字段等。
      通过这种规律可以改变关键词和翻页数字，获取图片URL页面源代码

10.1.2 数据预处理---去除无效和不相干图片
      下载图片数量没那么多:
          原因:
              1.搜索引擎返回的URL不一定都是有效结果
              2.搜索引擎返回结果的相关性随着排序的靠后越来越低，会出现很多不相关图片。
      通过Python，OpenCv尝试打开，打不开的就删除文件

10.1.3 数据预处理---去除重复图片
      训练时需要一部分作为验证集
      两种去重方法 : fdupes和findimagedupes

      fdupes: 通过MD5来查找并删除重复文件的工具，比较策略是先找到大小相同的文件，
              然后在结果里找到MD5相同的文件，然后逐byte比较
          安装    sudo apt install fdupes
          在数据所在文件夹下执行    fdupes  -rdN  ./
                  r代表递归查找，d代表删除重复图片并保留一张，N代表不确认直接删除
      findimagedupes:
          安装
          先建立一个train文件夹，把所有包含图片样本文件夹移入train文件夹下
                  mkdir train
                  mv  00? train
          减低样本分辨率      downscale.py文件
          统一缩放到短边不超过256像素
                  python downscale.py train 256
          图片去重
                  findimagedupes -R train > dup_list
          重复文件路径保存到dup_list文件里。

10.1.4 生成训练数据

        1. 从每类中采样一部分作为验证集 sample_val.py
        2. 增加小工具  link_data_agumentation.sh
        3. 执行run_augmentation.py
        4. 执行gen_label_list.py生成lmdb，产生文件和对应标签列表
        5. 执行 python gen_label_list.py train
                python gen_label_list.py val
            生成lmdb

10.2.1  迁移学习
        根据不同的物体推测其他物体
        One-shot
        Zero-Shot
        1.学习过程:
            第一部分:只关注模型学习各种分布之间的共性
            第二部分:具体任务，具体要区分的更细化的数据分布
        2.迁移学习能力来自分布式表征。
            特征是一层层组合起来的，越是底层的特征越基本，共性也越大。
            纹理特征:  底层的卷积核学到的是边缘，块等特征
            语义特征： 再高层学到的形状，顶层附近学习到的特征可以大致描述一个物体

10.2.2 模型微调法(Finetune)
        Caffe微调模型做法:
            1. 训练开始时用一个别人已经在大量数据集上训练好的模型和参数作为起始点
            2. 固定前面层的参数不动，只让后面一层或几层的参数在少量数据上进行学习
        训练好的模型在BVLC
        主要改动数据层和最后全连接层food_resnet_10_finetune_val.prototxt
            数据层换成自己的lmdb数据，
            最后全连接层输出改为7输出的fc_food,对计算精度和loss层做出相应改动
            输入前加入Dropout层，主要是因为数据量不多，为了防止过拟合

        Caffe的BatchNorm层
            1.对Scale层进行梯度下降
                  因为在caffe中BatchNorm层只实现了归一化步骤，最后缩放方差和偏置的步骤是经过再接一层Scale层完成
            2.BatchNorm层使用时需要把所有lr_mult置0
                  参数use_global_stats意思是是否使用全局的归一化参数
            3.原理:
                  在训练阶段，使用每个batch的方差和均值，预测阶段需要用全局。

        尝试自适应梯度下降法。AdaDelta，Solver描述:solver.prototxt

        执行python recognize_food.py val.txt结果保存一份在val_results.txt中

10.2.3 混淆举证(confusion_matrix)

        1. 以前评估模型时用准确率，准确率时用来评估模型的全局准确程度，只是数字，不包含太多细节信息
        2. 混淆矩阵，
              1. 把每个类别下，模型预测错误的结果数量，错误预测的类别和正确预测的数量在一个矩阵中显示出来啊
              2. 第三方库sklearn的metrics模块
                  sudo apt install python-sklearn
                  sudo pip install sklearn
        3. 在2100张测试集上得到的混淆矩阵和准确率
              make_confusion_matrix.py

        4.  混淆矩阵:
              1. 横轴是模型预测的类别数量统计
              2. 纵轴是数据真实标签的数量统计
              3. 对角线表示模型预测和数据标签一致的数目，
                  对角线之和除以测试集总数是准确率

10.2.4   P-R曲线和ROC曲线

         1. 二分类混淆矩阵
              PR(Precision-Recall)精度-召回曲线
              ROC(Receiver Operating Characteristic)受试者工作特征曲线
            指标:
                  1. Accuracy 准确率
                  2. Sensitivity  敏感度，即召回(Recall)或TRP(True Positive Rate)
                      预测为Positive的样本中正确的数量除以真正的Positive数量
                  3. 预测精度的PPV(Positive Predictive Value)

         2. 精度-召回曲线(Precision-Recall curve) 和 F分数(F-Measure)
              上面结果分析:一个假设,类别的判断是根据每个类别的概率最大值确定
              sort_kaoya_by_prob.py

         3. 受试者工作特征曲线(Receiver Operating Characteristic curve)
              ROC曲线啊通过选取不同阙值
              kaoya_shuizhurou_roc_auc.py
