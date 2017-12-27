import os
import random

每类采样300张
N = 300

建立val文件夹
os.system('mkdir -p val')

列出所有类别文件夹
class_dirs = os.listdir('train')

for class_dir in class_dirs:

    在val文件夹下建立对应类别文件夹
    os.system('mkdir -p val/{}'.format(class_dir))
    root = 'train/{}'.format(class_dir)
    print('Sampling validation set with {} images from {} ...'.format(N, root))
    filenames = os.listdir(root)

    对文件名进行乱序实现随机采样
    random.shuffle(filenames)

    取前300张
    val_filenames = filenames[:N]

    采样并移动文件到val文件夹下
    for filename in val_filenames:
        src_filepath = os.sep.join([root, filename])
        dst_filepath = os.sep.join(['val', class_dir, filename])
        cmd = 'mv {} {}'.format(src_filepath, dst_filepath)
        os.system(cmd)


执行代码得到验证集
