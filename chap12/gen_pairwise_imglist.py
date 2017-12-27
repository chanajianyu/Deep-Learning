import os
import random
import re

指定训练图片和验证图片的路径
train_dir = 'mnist/train'
val_dir = 'mnist/val'

指定样本对数目
n_train = 100000
n_val = 10000

从文件名中获取数字的正则表达式
pattern = re.compile('\d+_(\d)\.jpg')

生成训练集和验证集列表的文件
for img_dir, n_pairs in zip([train_dir, val_dir], [n_train, n_val]):

    列出所有文件名
    imglist = os.listdir(img_dir)

    获得所有样本数量
    n_samples = len(imglist)

    数据集名称
    dataset = img_dir[img_dir.rfind(os.sep)+1:]

    同时打开两个文件分别写入对应的文件列表
    with open('{}.txt'.format(dataset), 'w') as f, \
            open('{}_p.txt'.format(dataset), 'w') as f_p:

        按照指定数目随机写入文件路径对
        for i in range(n_pairs):

            随机选取第一个文件
            filename = imglist[random.randint(0, n_samples-1)]
            digit = pattern.findall(filename)[0]
            filepath = os.sep.join([img_dir, filename])

            随机选取第二个文件
            filename_p = imglist[random.randint(0, n_samples-1)]
            digit_p = pattern.findall(filename_p)[0]
            filepath_p = os.sep.join([img_dir, filename_p])

            如果两个文件是同一个数字则标签是1，否则标签是0
            label = 1 if digit == digit_p else 0

            分别写入两个文件
            f.write('{} {}\n'.format(filepath, label))
            f_p.write('{} {}\n'.format(filepath_p, label))
