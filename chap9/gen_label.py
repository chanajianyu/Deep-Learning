import os


取文件名不包含后缀部分，用下划线分割后两个字段
filename2score = lambda x: x[:x.rfind('.')].split('_')[-2:]

列出samples下所有图片文件名
filenames = os.listdir('samples')

创建训练集列表
with open('train.txt', 'w') as f_train_txt:
    for filename in filenames[:50000]:
        fx, fy = filename2score(filename)
        line = 'samples/{} {} {}\n'.format(filename, fx, fy)
        f_train_txt.write(line)

创建验证集列表
with open('val.txt', 'w') as f_val_txt:
    for filename in filenames[50000:60000]:
        fx, fy = filename2score(filename)
        line = 'samples/{} {} {}\n'.format(filename, fx, fy)
        f_val_txt.write(line)

测试集列表
测试集只有文件路径列表没有标签
with open('test.txt', 'w') as f_test_txt:
    for filename in filenames[60000:]:
        line = 'samples/{}\n'.format(filename)
        f_test_txt.write(line)


执行后，得到train.txt , val.txt , test.txt包含文件路径和对应频率的值 
