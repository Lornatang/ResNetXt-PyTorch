# ResNetXt-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

## Table of contents

- [ResNetXt-PyTorch](#resnetxt-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Aggregated Residual Transformations for Deep Neural Networks](#aggregated-residual-transformations-for-deep-neural-networks)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `resnetxt50_32x4d`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/ResNetXt50_32x4d-ImageNet_1K-7a64b822.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `resnetxt50_32x4d`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change
  to `./results/pretrained_models/ResNetXt50_32x4d-ImageNet_1K-7a64b822.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `resnetxt50_32x4d`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/resnetxt50_32x4d-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1611.05431.pdf](https://arxiv.org/pdf/1611.05431.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model       |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:-----------------:|:-----------:|:-----------------:|:-----------------:|
| resnetxt50_32x4d  | ImageNet_1K |   -(**19.13%**)   |   -(**4.69%**)    |
| resnetxt101_32x8d | ImageNet_1K |   -(**17.57%**)   |   -(**3.87%**)    |
| resnetxt101_64x4d | ImageNet_1K | 20.4%(**17.03%**) |  5.3%(**3.74%**)  |

```bash
# Download `ResNetXt50_32x4d-ImageNet_1K-7a64b822.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `resnetxt50_32x4d` model successfully.
Load `resnetxt50_32x4d` model weights `/ResNetXt-PyTorch/results/pretrained_models/ResNetXt50_32x4d-ImageNet_1K-7a64b822.pth.tar` successfully.
tench, Tinca tinca                                                          (80.20%)
barracouta, snoek                                                           (1.36%)
water bottle                                                                (0.16%)
armadillo                                                                   (0.09%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.08%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Aggregated Residual Transformations for Deep Neural Networks

*Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He*

##### Abstract

We present a simple, highly modularized network architecture for image classification. Our network is constructed by
repeating a building block that aggregates a set of transformations with the same topology. Our simple design results in
a homogeneous, multi-branch architecture that has only a few hyper-parameters to set. This strategy exposes a new
dimension, which we call "cardinality" (the size of the set of transformations), as an essential factor in addition to
the dimensions of depth and width. On the ImageNet-1K dataset, we empirically show that even under the restricted
condition of maintaining complexity, increasing cardinality is able to improve classification accuracy. Moreover,
increasing cardinality is more effective than going deeper or wider when we increase the capacity. Our models, named
ResNeXt, are the foundations of our entry to the ILSVRC 2016 classification task in which we secured 2nd place. We
further investigate ResNeXt on an ImageNet-5K set and the COCO detection set, also showing better results than its
ResNet counterpart. The code and models are publicly available online.

[[Paper]](https://arxiv.org/pdf/1611.05431.pdf)

```bibtex
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}
```