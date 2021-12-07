## tutils
(tutils.py)
tfilename
```
tfilename('x', 'x', 'x') = os.path.join() + makedirs
```



## Visualizer
(visualizer)
TSNE
torchvision.utils.save_image

## Logger
MetricLogger (metriclogger.py)

配置管理 + Logger  (initializer.py, tlogger.py)
```
parser = argparse.ArgumentParser(description='')
args = trans_args(parser)
logger, config = trans_init(args)
```

#### Basic Usasge

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

#### MultiLogger

Set mode = ['tensorboard', 'wandb']

```
logger.add_scaler(key, value, global_step=-1)
```