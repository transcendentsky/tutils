# Augment
`timg`
- compress_JPEG_from_path(path, quality=10)
    - JPEG image compressing
- gaussian_blur(img, kernel=(3, 3))
- class Augment
- partial_augment(img, augment)


# Printing
`print_img`
- print_img_auto(img, img_type='ori', is_gt=True, fname=None)
    - type="img"/"exr"/"bg" ...
    - save_img by cv2

`draw_heatmap`
- draw_heatmap(points: np.ndarray, points2, fname="testtt.png")
- draw_scatter(points, points2, fname="ttest.png", c="red")


# Timer
`tuils.timer`
- tenum() -> return time, i , res
- tfunctime -> @tfunctime -> return time, res

# Logger
`tutils.tlogger`
- trans_init() -> return logger, config, tag, runs_dir
- logger.info(...) -> [time @x.py] INFO ...

# ------------------------------------------------------
# Notes

## Three main parts:

* tutils:
  * MultiLogger:
    * logging logger
    * Wandb logger
    * Tensorboard logger
  * Tools:
    * tfilename / tdir / tenum / timer

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
  - trainer:

    - usage:``trainer = Trainer(logger=logger, config=config)``
  - recorder:

    - Recorder
    - CSVLogger
    - ExcelLogger
  - tester
- visualize

# Tutils logger usage

- trans_args: return args containing 'tag', 'extag', 'config'
- trans_init: integrate all configs and return to trans_configure
- trans_configure: set logger, runs dir and config...

## Basic Usasge

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

# How to install

```
pip install git+http://gitee.com/transcendentsky/tutils.git
```

# TODO:

Code checker:
dataloader checking
runtime checking

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

# Upload to Pypi
python setup.py bdist_wheel
python -m twine upload dist/*
