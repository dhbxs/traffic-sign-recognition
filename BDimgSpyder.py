import requests
import re
import os
import time

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'}
name = input('您要爬取什么图片')
num = 0
x = input('您要爬取几张呢?，输入1等于60张图片。')
for i in range(int(x)):
    name_1 = "D:\\猫\\"
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='+name+'&pn='+str(i*30)
    res = requests.get(url,headers=headers)
    htlm_1 = res.content.decode()
    a = re.findall('"objURL":"(.*?)",',htlm_1)
    if not os.path.exists(name_1):
        os.makedirs(name_1)
    for b in a:
        num = num +1
        try:
            img = requests.get(b)
        except Exception as e:
            print('第'+str(num)+'张图片无法下载------------')
            print(str(e))
            continue
        f = open(name_1+name+str(num)+'.jpg','ab')
        print('---------正在下载第'+str(num)+'张图片----------')
        f.write(img.content)
        f.close()
print('下载完成')