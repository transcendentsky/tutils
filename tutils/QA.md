
## Writing Tools
Tools for recognizing symbols, 识别各种符号
Detexify:
https://detexify.kirelabs.org/classify.html

Tables Generator， 为Latex自动生成表格
https://www.tablesgenerator.com/

Formula Recognizer, or generate formula for Latex, 识别公式，可以转换成Latex格式 （Ps: 有使用次数，超过收费）
https://mathpix.com/

Draw.io, 画图网站
https://app.diagrams.net/



## Change Python Source:

阿里云 http://mirrors.aliyun.com/pypi/simple/
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
豆瓣(douban) http://pypi.douban.com/simple/
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

#
linux:
修改 ~/.pip/pip.conf (没有就创建一个)， 内容如下：

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

#
windows:
直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini，内容如下

```
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```
