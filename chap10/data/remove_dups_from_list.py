import os
import sys

读取文件列表，保留每行第一个文件，删除其余文件
dup_list = sys.argv[1]

with open(dup_list, 'r') as f:
    lines = f.readlines()
    for line in lines:

        列出所有重复图像文件
        dups = line.split()
        print('Removing duplicates of {}'.format(dups[0]))

        保留第一张，其他删除
        for dup in dups[1:]:
            cmd = 'rm {}'.format(dup)
            os.system(cmd)
