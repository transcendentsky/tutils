**Py-Tutorials (example)**
======== 
<!-- Description -->
![Trans](./corgi1.jpg)

Paper link: arxiv.org/xxx

Abstract: There are some demo codes and test codes
一些python各个功能的示例, 以及测试代码

# Usage

## Environment
python == 3.5/3.6, 
pytorch >= 1.1.0, 
torchvison >= 0.6


```
pip install -r requirements.txt
```

## Data preparation
We train/test our model on Datasets (e.g. [KiTS19](http://xxx.org/xxx/xx) ) 
<!-- or You can download from link: xxx -->

We expect the directory structure to be the following:
```
path/to/kits19
    data/
        case_00xxx/
            imaging.nii.gz
            segmentation.nii.gz
        case_00xxx/
        ...
```
## Training
Pre-processing
```
nii2pickle xxx
```
To train our model, run this scripts
```
python -m scripts.train --epochs 300 --data_path path/to/kits19 
```
To evaluate our model, run this scripts
```
python -m scripts.test --data_path path/to/kits19 --resume xxx-model.pth
```

# Citation
Please cite our paper if it helps you.
```
@proceeding{
    title
}
```
# License
This code is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.

# Contribution
We actively welcome your pull requests! feel free!
