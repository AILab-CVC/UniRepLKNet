## UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video,Point Cloud, Time-Series and Image Recognition

<p align="center" width="100%">
<img src="assets/banner.png"  width="100%" height="60%">
</p>


<div align="center">
    <span class="author-block">
    <a href="https://dingxiaohan.xyz/" target="_blank">Xiaohan Ding</a><sup>1*</sup>,
    </span>
    <span class="author-block">
    <a href="https://invictus717.github.io/" target="_blank">Yiyuan Zhang</a><sup>2*</sup>,</span>
    <span class="author-block">
    </span>
    <a href="https://geyixiao.com/" target="_blank">Yixiao Ge</a><sup>1</sup>,
    </span>
    </br>
    <span class="author-block">
    <a target="_blank">Sijie Zhao</a><sup>1</sup>,
    </span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=6Ra2TgQAAAAJ&hl=en&oi=ao" target="_blank">Lin Song</a><sup>1</sup>,
    </span>
    <span class="author-block">
    <a href="http://people.eecs.berkeley.edu/~xyyue/" target="_blank">Xiangyu Yue</a><sup>2</sup>,
    </span>
    <span class="author-block">
    <a href="https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en&oi=ao" target="_blank">Ying Shan</a><sup>1</sup>
    </span>

</div>

<div align="center">
    <sup>1</sup> <a href='https://ai.tencent.com/' target='_blank'>Tencent AI Lab</a>
    <sup>2</sup>
    <a href='http://mmlab.ie.cuhk.edu.hk/' target='_blank'>The Chinese University of Hong Kong</a>&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
</div>

[![arXiv](https://img.shields.io/badge/arxiv-2311.15599-b31b1b?style=plastic&color=b31b1b&link=https%3A%2F%2Farxiv.org%2Fabs%2F2311.15599)](https://arxiv.org/abs/2311.15599)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main)
[![website](https://img.shields.io/badge/Project-Website-blueviolet)](https://invictus717.github.io/UniRepLKNet/)
<a href="#LICENSE--citation">
  <img alt="License: Apache2.0" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue.svg"/>
</a>

## Motivation 
* We note that most architectures of the existing large-kernel ConvNets simply follow other models. **The architectural design for large-kernel ConvNets remains under-explored.**
* The *universal perception ability* of Transformers is sparking in multimodal research areas (image, audio, video, time-series, *etc*). We are curious whether ConvNets can also deliver **universal perception ability across multiple modalities with a unified architecture**.

## Highlights

In this paper, we contribute from two aspects:
* We propose four architectural guidelines for designing
large-kernel ConvNets, the core of which is to exploit
the essential characteristics of large kernels that distinguish
them from small kernels - they can see wide without
going deep. Following such guidelines, our proposed
large-kernel ConvNet shows leading performance in image
recognition. For example, our models achieve an ImageNet
accuracy of 88.0%, ADE20K mIoU of 55.6%, and COCO
box AP of 56.4%, demonstrating better performance and
higher speed than a number of recently proposed powerful
competitors. 
* We discover that large kernels are the key to
unlocking the exceptional performance of ConvNets in domains
where they were originally not proficient. With certain
modality-related preprocessing approaches, the proposed
model achieves state-of-the-art performance on time-series
forecasting and audio recognition tasks even without
modality-specific customization to the architecture.

**UniRepLKNet not only signifies a "comeback" for ConvNet in its original domain but also showcases large-kernel ConvNet’s potential to "conquer" new territories, highlighting further adaptability and broad utility across different modalities and tasks.**

## TODOs

- [x] Model code
- [x] Most of the ImageNet-1K and ImageNet-22K pretrained weights
- [x] Weights released on both Google Drive (see this page) and hugging face (see unireplknet.py)
- [x] PyTorch efficient large-kernel conv implementation
- [x] ImageNet training code
- [x] Code, models, and documents of audio, video, point cloud, and time-series tasks (will be released in one day)
- [ ] Better documentation
- [ ] Object detection and semantic segmentation code and models (will be released in one day)


**Star and watch me if you are interested in this project :)**

**There may be some bugs. Please raise an issue if you get one. The code will be thoroughly tested in the next several days.**

## Models

We have uploaded the weights to Google Drive. You may alternatively download via hugging face (see [HERE](https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py#L675)).

### ImageNet-1K Pretrained Weights

| name | resolution |acc@1 | #params | FLOPs | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|
| UniRepLKNet-A | 224x224 | 77.0 | 4.4M  | 0.6G | [ckpt](https://drive.google.com/file/d/1jUB-lq6NMTbeBvGTDvAarKWh-ZfMMZWt/view?usp=drive_link) |
| UniRepLKNet-F | 224x224 | 78.6 | 6.2M  | 0.9G | [ckpt](https://drive.google.com/file/d/1vYqhCNx3q-z0fVT4UZecFTUmb9IDaYh9/view?usp=drive_link) |
| UniRepLKNet-P | 224x224 | 80.2 | 10.7M  | 1.6G | [ckpt](https://drive.google.com/file/d/1D7rljWnnzEGDn8MDkvAWJ8qd1SCix6Vm/view?usp=drive_link) |
| UniRepLKNet-N | 224x224 | 81.6 | 18.3M | 2.8G | [ckpt](https://drive.google.com/file/d/1tMHOl55C7h44ag8SLUuaP0bBUUpVXhKj/view?usp=drive_link) |
| UniRepLKNet-T | 224x224 | 83.2 | 31M | 4.9G | [ckpt](https://drive.google.com/file/d/12Xon3FWkzEQV1nnNsF2U8XDMD-7NO2cJ/view?usp=drive_link) |
| UniRepLKNet-S | 224x224 | 83.9 | 56M   | 9.1G | [ckpt](https://drive.google.com/file/d/11YEOcKs4WNprRzCvKe-fB5z-l7zQv3kb/view?usp=drive_link) |

### ImageNet-22K Pretrained Weights

| name | resolution | #params | FLOPs | ckpt |
|:---:|:---:|:---:|:---:| :---:|
| UniRepLKNet-S | 224x224 | 56M | 26.7G  | TBA |
| UniRepLKNet-B | 224x224 | 98M   | 47.2G   | [ckpt](https://drive.google.com/file/d/1t1txZOTpwXGUMVsqyUxpzE5EGLqMX5li/view?usp=drive_link)|
| UniRepLKNet-L | 192x192 | 218M  | 105.4G   | [ckpt](https://drive.google.com/file/d/1PEY474n6a7pZ3vJitsU7ZLzwBI00pf7u/view?usp=drive_link)|
| UniRepLKNet-XL | 192x192 | 386M  | 187G  | [ckpt](https://drive.google.com/file/d/1OP7I0jabljm8LKXTypk4HDmF9dQQqYib/view?usp=drive_link)|

### Pretrained on ImageNet-22K then finetuned on ImageNet-1K

| name | resolution |acc@1 | #params | FLOPs | ckpt |
|:---:|:---:|:---:|:---:| :---:| :---:|
| UniRepLKNet-S | 384x384 | 86.4 | 56M | 26.7G  | [ckpt](https://drive.google.com/file/d/1PzEHFOgEllMRIB-emkX_2VjXyBYC_X0z/view?usp=drive_link)|
| UniRepLKNet-B | 384x384 | 87.4 | 98M   | 47.2G   | [ckpt](https://drive.google.com/file/d/1T4BB3xx6FsWrK5QpTy7FwBrLuOMcZcEu/view?usp=drive_link)|
| UniRepLKNet-L | 384x384 | 87.9 | 218M  | 105.4G   | [ckpt](https://drive.google.com/file/d/10jJGzXX3cFRrfk3oAoIoWRnKSAaquQtM/view?usp=drive_link)|
| UniRepLKNet-XL | 384x384 | 88.0 | 386M  | 187G  | TBA|

### COCO Object Detection

Code, weights and configs will be released in one day.

| name | resolution |box mAP | mask mAP | #params | FLOPs | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| UniRepLKNet-T | 1280x800 | 51.7 | 44.9 | 89M  | 749G | TBA |
| UniRepLKNet-S | 1280x800 | 53.0 | 45.9 | 113M  | 835G | TBA |
| UniRepLKNet-S_22K | 1280x800 | 54.3 | 47.1 | 113M  | 835G | TBA |
| UniRepLKNet-B_22K | 1280x800 | 54.8 | 47.4 | 155M  | 978G | TBA |
| UniRepLKNet-L_22K | 1280x800 | 55.8 | 48.4 | 276M  | 1385G | TBA |
| UniRepLKNet-XL_22K | 1280x800 | 56.4 | 49.0 | 443M  | 1952G | TBA |

### ADE-20K Semantic Segmentation

Code, weights and configs will be released in one day.

| name | resolution |mIoU (ss/ms) | #params | FLOPs | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|
| UniRepLKNet-T | 512x512 | 48.6/49.1 | 61M | 946G  | TBA |
| UniRepLKNet-S | 512x512 | 50.5/51.0 | 86M  | 1036G | TBA |
| UniRepLKNet-S_22K | 640x640 | 51.9/52.7 | 86M  | 1036G | TBA |
| UniRepLKNet-B_22K | 640x640 | 53.5/53.9 | 130M  | 1850G | TBA |
| UniRepLKNet-L_22K | 640x640 | 54.5/55.0 | 254M  | 2507G | TBA |
| UniRepLKNet-XL_22K | 640x640 | 55.2/55.6 | 425M  | 3420G | TBA |

## ImageNet evaluation and training

We give an example evaluation command.

Single-GPU
```
python main.py --model unireplknet_b --eval true \
--resume unireplknet_b_in22k_to_in1k_384_acc87.40.pth  \
--input_size 384 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model unireplknet_b --eval true \
--resume unireplknet_b_in22k_to_in1k_384_acc87.40.pth  \
--input_size 384 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
For training or finetuning UniRepLKNets on ImageNet-1K or 22K, see [THIS DOC](/Image/README.md)

## Universal perception on audio, video, point cloud, and time-series tasks

For detailed documetation, please refer to these documents as follows:

* Audio for [Audio DOC](/Audio/README.md)
* Point Cloud [Point Cloud DOC](/Point/README.md)
* Time-Series [Time-Series DOC](/Time-Series/README.md)
* Video [Video DOC](/Video/README.md)

## Use an efficient large-kernel convolution with PyTorch

We use a large-kernel conv implementation in **PyTorch** that is more efficient than the native torch.nn.Conv2d . It is implemented based on the iGEMM algorithm. Please check ```setup.py``` and ```depthwise_conv2d_implicit_gemm.py``` (**a replacement of torch.nn.Conv2d**) in https://github.com/MegEngine/cutlass/tree/master/examples/19_large_depthwise_conv2d_torch_extension.

1. ```unzip cutlass.zip```, enter the directory. This version of cutlass works fine with our large-kernel implementation and multiple python versions. You may alternatively use the cutlass branch maintained by the MegEngine team (clone https://github.com/MegEngine/cutlass), but you may need to be more careful with your python version (see [this issue](https://github.com/DingXiaoH/RepLKNet-pytorch/issues/34)).
2. ```cd examples/19_large_depthwise_conv2d_torch_extension```
3. ```./setup.py install --user```. If you get errors, check your ```CUDA_HOME```.
4. A quick check: ```python depthwise_conv2d_implicit_gemm.py```
5. Add ```WHERE_YOU_CLONED_CUTLASS/examples/19_large_depthwise_conv2d_torch_extension``` into your ```PYTHONPATH``` so that you can ```from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM``` anywhere. Then you may use ```DepthWiseConv2dImplicitGEMM``` as a replacement of ```nn.Conv2d```.
6. If you do not install this implementation, you can still use our model anywhere you wish but it will be a bit slower.

It should work with a wide range of GPUs and PyTorch/CUDA versions. We suggest you try first and check the environments only if you get any errors. Our latest testes used both

1. Ubuntu 18.04 + CUDA 11.3 + nvcc 11.3 + cudnn 8.2.0 + python 3.8.12 + pytorch 1.10 + gcc 7.3.0 + nccl 2.10.3 + NVIDIA driver 450.102.04 + V100 and A100 GPUs
2. Ubuntu 18.04 + CUDA 10.2 + nvcc 10.0 + cudnn 7.6.5 + python 3.6.9 + pytorch 1.9 + gcc 7.5.0 + nccl 2.7.8 + NVIDIA driver 460.32.03 + 2080Ti and V100 GPUs

It is reported (see [here](https://github.com/DingXiaoH/RepLKNet-pytorch/issues/34)) that a python version mismatch may result in an error (```forward_fp32.cu(212): error: more than one instance of constructor "cutlass::Tensor4DCoord::Tensor4DCoord" ...``` or ```cutlass/include/cutlass/fast_math.h(741): error: no suitable conversion function from "__half" to "float" exists```). Please upgrade or downgrade your python. We sincerely thank @sleeplessai and @ewrfcas for sharing their experience.

Pull requests (e.g., better or other implementations or implementations on other frameworks) are welcomed.

## Citation

If the code and paper help your research, please kindly cite:

```
@article{ding2023unireplknet,
  title={UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition},
  author={Ding, Xiaohan and Zhang, Yiyuan and Ge, Yixiao and Zhao, Sijie and Song, Lin and Yue, Xiangyu and Shan, Ying},
  journal={arXiv preprint arXiv:2311.15599},
  year={2023}
}
```
## License
This project is released under the Apache 2.0 license. Please see the [LICENSE](LICENSE) file for more information.
