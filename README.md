# Notes

## Three main parts:

* tutils: (收集各种工具)
  * MultiLogger:
    * logging logger
    * Wandb logger
    * Tensorboard logger
    - CSVLogger
    - ExcelLogger
  * Tools:
    * tfilename / tdir / tenum / timer
  - visualizer
    - tsne
    - save_image(tensor)
- mn (收集模型/模块)
  - data: pre-process operations for data, and augmentations
  
    - augment
    - preprocess
    - SimpleITK ops (medical images reading / saving)
  - train:  
    - models
    - loss
    - TODOs:
      - optimizers
      - schedulers
  - eval:
  
    - metrics
    - radiology analysis
- trainer (自定义训练框架 - 参考 pytorch-lightning )
  - learner
  - trainer:

    - usage:``trainer = Trainer(logger=logger, config=config)``
  - recorder:

    - Recorder
  - tester

# How to install

```
pip install trans-utils
or
pip install git+http://gitee.com/transcendentsky/tutils.git
```

# TODO:

Code checker:
dataloader checking
runtime checking
