# Notes

## Three main parts:
- data: pre-process operations for data, and augmentations
    - augment
    - preprocess
    - SimpleITK ops (medical images reading / saving)
- train:
    - models
    - loss
    - TODO:
        - optimizers
        - schedulers
- eval:
    - metrics
    - radiology analysis

- framework
    - learner
    - trainer
    - tester
    
- visualize

# Tutils logger usage
- trans_args: return args containing 'tag', 'extag', 'config'
- trans_init: integrate all configs and return to trans_configure
- trans_configure: set logger, runs dir and config...

## basic usasge
```
args = trans_args() or args = trans_args(parser) with custom parser
```
then
```
logger, config = trans_init(args)
```
if you want to use your own configs, you can also try
```
logger, config = trans_configure(config)
```

## MultiLogger 
Set mode = ['tensorboard', 'wandb']
```
logger.add_scaler(key, value, global_step=-1)
```


## How to install
pip install git+http://gitee.com/transcendentsky/tutils.git

#
# TODO:
Code checker: 
    dataloader checking
    runtime checking
    












## python Source:

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