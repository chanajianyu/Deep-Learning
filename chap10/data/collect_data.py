import os
import re
import urllib
from multiprocessing import Process

支持JPG和PNG格式
SUPPORTED_FORMATS = ['jpg', 'png', 'jpeg']

搜索引擎获取关键字和指定样本的模块
{keyword}是关键词字段
{index}是图片开始下标字段
URL_TEMPLATE = r'http://image.b***u.com/search/flip?tn=b***uimage&ie=utf-8&word={keyword}&pn={index}'

定义每个进程内下载的函数，参数分别是:
    dir_name：文件要保存的位置
    keyword: 关键词
    start_index: 要下载文件的开始编号
    end_index: 要下载文件的结束编号

def download_images_from_b***u(dir_name, keyword, start_index, end_index):
    index = start_index

    结束下载的判断
    while index < end_index:

        通过输入关键字和当前下标生成获取返回结果的URL
        url = URL_TEMPLATE.format(keyword=keyword, index=index)
        try:

            打开URL获取HTML源码
            html_text = urllib.urlopen(url).read().decode('utf-8', 'ignore')

            正则匹配获取所有URL
            image_urls = re.findall(r'"objURL":"(.*?)"', html_text)

            很可能没有更多返回结果，结束当前函数
            if not image_urls:
                print('Cannot retrieve anymore image urls \nStopping ...'.format(url))
                break

        如果发生I/O错误，可能是没有返回结果，或者超时严重，结束当前函数
        except IOError as e:
            print(e)
            print('Cannot open {}. \nStopping ...'.format(url))
            break

        记录已经下载的URL
        downloaded_urls = []

        依次下载URL对应图片
        for url in image_urls:
            filename = url.split('/')[-1]
            ext = filename[filename.rfind('.')+1:]

            URL命名复杂，只下载支持的文件后缀
            if ext.lower() not in SUPPORTED_FORMATS:
                index += 1
                continue

            为了方便后期处理，直接在下载阶段指定好要保存的文件名为数字的规则形式
            filename = '{}/{:0>6d}.{}'.format(dir_name, index, ext)
            cmd = 'wget "{}" -t 3 -T 5 -O {}'.format(url, filename)
            os.system(cmd)

            如果下载成功，并且文件大于1kb，说明是个有效的图片文件
            if os.path.exists(filename) and os.path.getsize(filename) > 1024:
                index_url = '{:0>6d},{}'.format(index, url)
                downloaded_urls.append(index_url)
            else:
                否则是无效文件，删除
                os.system('rm {}'.format(filename))

            判断是否已经下载完当前进程的任务
            index += 1
            if index >= end_index:
                break

        保存已经下载的图片并和文件名建立对应关系
        with open('{}_urls.txt'.format(dir_name), 'a') as furls:
            urls_text = '{}\n'.format('\n'.join(downloaded_urls))
            if len(urls_text) > 11:
                furls.write(urls_text)

启动下载任务函数
def download_images(keywords, num_per_kw, procs_per_kw):
    args_list = []

    生成每个进程子任务的输入参数列表
    for class_id, keyword in enumerate(keywords):
        dir_name = '{:0>3d}'.format(class_id)
        os.system('mkdir -p {}'.format(dir_name))
        num_per_proc = int(round(float(num_per_kw/procs_per_kw)))
        for i in range(procs_per_kw):
            start_index = i * num_per_proc
            end_index = start_index + num_per_proc - 1
            args_list.append((dir_name, keyword, start_index, end_index))

    生成进程列表
    processes = [Process(target=download_images_from_b***u, args=x) for x in args_list]

    开始并行下载
    print('Starting to download images with {} processes ...'.format(len(processes)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print('Done!')

if __name__ == "__main__":
    读取关键词列表
    with open('keywords.txt', 'rb') as f:
        foods = f.read().split()

    设置每个类别下载目标2000张，每类别用3个进程下载
    download_images(foods, 2000, 3)
