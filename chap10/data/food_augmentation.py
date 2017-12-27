import os

每类增加到3000张
n_total = 3000

从train文件夹获取子文件夹
class_dirs = os.listdir('train')

遍历每个子文件夹
for class_dir in class_dirs:
    src_path = 'train/{}'.format(class_dir)

    获取类别下已有样本数量
    n_samples = len(os.listdir(src_path))

    计算需要增加的样本数量
    n_aug = n_total - n_samples

    执行数据增加，并在temp文件夹下生成增加数据
    cmd = 'python run_augmentation.py {} temp {}'.format(src_path, n_aug)
    os.system(cmd)

    把增加的数据添加到原文件夹下
    cmd = 'mv temp/* {}'.format(src_path)
    os.system(cmd)
    
删除临时文件夹
os.system('rm -r temp')
