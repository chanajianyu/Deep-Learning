import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

图像大小
IMAGE_SIZE = (100, 100)

减均值帮助收敛
MEAN_VALUE = 128

获取输入图像和标签列表文件名
filename = sys.argv[1]

获取文件名不包含后缀部分作为数据集名称
setname, ext = filename.split('.')

读取所有行
with open(filename, 'r') as f:
    lines = f.readlines()

乱序
np.random.shuffle(lines)

样本数量
sample_size = len(lines)

图像数据格式，样本总量*通道数*高*宽
imgs = np.zeros((sample_size, 1,) + IMAGE_SIZE, dtype=np.float32)
频率标签格式，样本总量*标签数
freqs = np.zeros((sample_size, 2), dtype=np.float32)

生成h5文件名
h5_filename = '{}.h5'.format(setname)

将数据写入h5文件，图像对应数据名称是data，标签对应数据名称是freq
with h5py.File(h5_filename, 'w') as h:
    for i, line in enumerate(lines):
        image_name, fx, fy = line[:-1].split()

        因为直接用plt.imread()读取的图片是三通道，所以只取第一个通道
        img = plt.imread(image_name)[:, :, 0].astype(np.float32)
        img = img.reshape((1, )+img.shape)
        img -= MEAN_VALUE
        imgs[i] = img
        freqs[i] = [float(fx), float(fy)]
        if (i+1) % 1000 == 0:
            print('Processed {} images!'.format(i+1))
    h.create_dataset('data', data=imgs)
    h.create_dataset('freq', data=freqs)

生成h5文件列表
with open('{}_h5.txt'.format(setname), 'w') as f:
    f.write(h5_filename)

将这段代码保存为gen_hdf5.py,然后执行以下命令:
> python gen_hdf5.py train.txt
> python gen_hdf5.py val.txt
得到训练集和验证集的h5文件及对应列表文件
