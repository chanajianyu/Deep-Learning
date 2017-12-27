import os
import sys
import datetime
import cv2

这段代码得到一个samples文件夹
    1. 随机得到一个1-100随机数，原始噪声的横轴和纵轴长度
    2. 按照这个大小产生0-1间均匀分布的随机浮点矩阵作为原始图像
    3. 把随机噪声原始图像放大100*100得到不同频率的噪声图像

from multiprocessing import Process, cpu_count

import numpy as np
import matplotlib.pyplot as plt


定义长宽和样本数量
H_IMG, W_IMG = 100, 100
SAMPLE_SIZE = 70000

保存样本的文件名
SAMPLES_DIR = 'samples'

产生一个噪声并保存到Simple文件夹
def make_noise(index):

    随机生成高和宽
    h = np.random.randint(1, H_IMG)
    w = np.random.randint(1, W_IMG)

    原始噪声图像
    noise = np.random.random((h, w))

    用cubic插值放大让图像平滑
    noisy_img = cv2.resize(noise, (H_IMG, W_IMG), interpolation=cv2.INTER_CUBIC)

    原始图像的宽高与要放大到的宽高比值即相对频率

    频率高低用一个0-1间的数字表示，0是完全没有变化，1表示每个像素之间是不相关的最高频率变化
    fx = float(w) / float(W_IMG)
    fy = float(h) / float(H_IMG)

    文件名包含3个字段:噪声序号，横轴频率，纵轴频率
    filename = '{}/{:0>5d}_{}_{}.jpg'.format(SAMPLES_DIR, index, fx, fy)
    plt.imsave(filename, noisy_img, cmap='gray')

def make_noises(i0, i1):
    产生下标i0到i1的噪声图像
    np.random.seed(datetime.datetime.now().microsecond)
    for i in xrange(i0, i1):
        make_noise(i)
    print('Noises from {} to {} are made!'.format(i0+1, i1))
    sys.stdout.flush()

def main():
    建立保存图像的文件夹
    cmd = 'mkdir -p {}'.format(SAMPLES_DIR)
    os.system(cmd)

    默认获取所有可用CPU核数作为多进程数量
    n_procs = cpu_count()

    print('Making noises with {} processes ...'.format(n_procs))

    产生尽量均匀的任务列表
    length = float(SAMPLE_SIZE)/float(n_procs)
    indices = [int(round(i * length)) for i in range(n_procs + 1)]

    定义每个进程
    processes = [Process(target=make_noises, args=(indices[i], indices[i+1])) for i in range(n_procs)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('Done!')

if __name__ == '__main__':
    main()
