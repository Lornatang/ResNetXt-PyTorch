# Wide_ResNet-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v4.pdf).

## Table of contents

- [Wide_ResNet-PyTorch](#wide_resnet-pytorch)
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
        - [Wide Residual Networks](#wide-residual-networks)

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

- line 29: `model_arch_name` change to `wide_resnet50`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/Wide_ResNet50-ImageNet_1K-d5b3452e.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `wide_resnet50`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change to `./results/pretrained_models/Wide_ResNet50-ImageNet_1K-d5b3452e.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `wide_resnet50`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/wide_resnet50-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1605.07146v4.pdf](https://arxiv.org/pdf/1605.07146v4.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|     Model      |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:--------------:|:-----------:|:-----------------:|:-----------------:|
| wide_resnet50  | ImageNet_1K | 21.9%(**18.70%**) | 6.03%(**4.49%**)  |
| wide_resnet101 | ImageNet_1K |  2-(**26.71%**)   |   -(**8.58%**)    |

```bash
# Download `Wide_ResNet50-ImageNet_1K-d5b3452e.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `wide_resnet50` model successfully.
Load `wide_resnet50` model weights `/Wide_ResNet-PyTorch/results/pretrained_models/Wide_ResNet50-ImageNet_1K-d5b3452e.pth.tar` successfully.
tench, Tinca tinca                                                          (59.80%)
barracouta, snoek                                                           (1.35%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.19%)
plastic bag                                                                 (0.18%)
water bottle                                                                (0.13%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Wide Residual Networks

*Sergey Zagoruyko, Nikos Komodakis*

##### Abstract

Deep residual networks were shown to be able to scale up to thousands of layers
and still have improving performance. However, each fraction of a percent of improved
accuracy costs nearly doubling the number of layers, and so training very deep residual networks has a problem of
diminishing feature reuse, which makes these networks
very slow to train. To tackle these problems, in this paper we conduct a detailed experimental study on the architecture
of ResNet blocks, based on which we propose a novel
architecture where we decrease depth and increase width of residual networks. We call
the resulting network structures wide residual networks (WRNs) and show that these are
far superior over their commonly used thin and very deep counterparts. For example,
we demonstrate that even a simple 16-layer-deep wide residual network outperforms in
accuracy and efficiency all previous deep residual networks, including thousand-layerdeep networks, achieving new
state-of-the-art results on CIFAR, SVHN, COCO, and
significant improvements on ImageNet. Our code and models are available
at https://github.com/szagoruyko/wide-residual-networks.

[[Paper]](https://arxiv.org/pdf/1605.07146v4.pdf)

```bibtex
@article{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}
```