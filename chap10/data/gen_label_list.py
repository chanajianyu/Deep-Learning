import os
import sys

获取输入文件的名称作为数据集名字
dataset = sys.argv[1].rstrip(os.sep)

class_dirs = os.listdir(dataset)

遍历每个类别并将生成的列表写入文件
with open('{}.txt'.format(dataset), 'w') as f:
    for class_dir in class_dirs:
        class_path = os.sep.join([dataset, class_dir])
        label = int(class_dir)
        lines = ['{}/{} {}'.format(class_path, x, label) for x in os.listdir(class_path)]
        f.write('\n'.join(lines) + '\n')
