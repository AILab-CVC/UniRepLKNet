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

**Star and watch me if you are interested in this project :)**

**There may be some bugs. Please raise an issue if you get one. The code will be thoroughly tested in the next several days.**

## Motivation 
* We note that most architectures of the existing large-kernel ConvNets simply follow other models. **The architectural design for large-kernel ConvNets remains under-explored.**
* The *universal perception ability* of Transformers is sparking in multimodal research areas (image, audio, video, time-series, *etc*). We are curious whether ConvNets can also deliver **universal perception ability across multiple modalities with a unified architecture**.

## Highlights

A **ConvNet unifies multiple modalities and outperforms modality-specific models**. This paper summarizes architectural guidelines to build large-kernel CNN, which works amazingly well with images and other modalities. This is the latest contribution to both two influential areas - **Structural Re-param** (since RepVGG, Ding et al. 2021) and **very-large-kernel ConvNet** (since RepLKNet, Ding et al. 2022). **ImageNet accuracy of 88.0%, COCO AP of 56.4, ADE20K mIoU of 55.6 with only ImageNet-22K pretraining**. Higher actual speed and performance than recent models like ConvNeXt v2 and InternImage. With a unified architecture and extremely simple modality-specific preprocessing, achieves state-of-the-art performances on audio recognition and, most amazingly, **Global Temperature & Wind Speed Forecasting** (a challenging huge-scale time-series forecasting task), outperforming the existing global forecasting system.

More specifically, we contribute from two aspects:
* We propose four architectural guidelines for designing
large-kernel ConvNets, the core of which is to exploit
the essential characteristics of large kernels that distinguish
them from small kernels - they can see wide without
going deep. Following such guidelines, our proposed
large-kernel ConvNet shows leading performance in image
recognition.  
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
- [x] Code and documents of audio, video, point cloud, and time-series tasks
- [x] Semantic segmentation code, document, and most checkpoints
- [x] Object detection code, document, and all the checkpoints
- [ ] Checkpoints of audio, video, point cloud, and time-series tasks

The ImageNet, COCO, and ADE20K checkpoints have been released (see the huggingface repo shown below), except the ImageNet-22K pretrained UniRepLKNet-S, and UperNet with UniRepLKNet-XL, which were lost, and we are reproducing them.

Latest news: fixed a bug, which results from [this commit](https://github.com/AILab-CVC/UniRepLKNet/commit/920b7251ea3d52ab476d0f40ba722db56d9a7e03) on Dec 1st, 2023. [Now it is fixed ](https://github.com/AILab-CVC/UniRepLKNet/commit/5349bcee9a8202c62c8c169220f8cc613914baac). If you used unireplknet.py after Dec 1st, 2023, please check your code.




## Code design

1. There is some MMDetection- and MMSegmentation-related code in ```unireplknet.py``` so that you can directly copy-paste it into your MMDetection or MMSegmentation, e.g., [here](unireplknet.py#L29) and [here](unireplknet.py#L617). If you do not want to use it with MMDetection or MMSegmentation, you can safely delete those lines of code.
2. We have provided code to automatically build our models and load our released weights. See the functions [here](unireplknet.py#L726). You can also use ```timm.create_model``` to build the models. For example, ```model = timm.create_model('unireplknet_l', num_classes=num_classes_of_your_task, in_22k_pretrained=True)``` will call the function ```unireplknet_l``` defined [here](https://github.com/AILab-CVC/UniRepLKNet/blob/main/unireplknet.py#L745), which will build a UniRepLKNet-L and automatically download our checkpoints and load the weights.
3. As UniRepLKNet also uses the Structural Re-parameterization methodology, we provide a function ```reparameterize_unireplknet()``` that converts a trained UniRepLKNet into the inference structure, which equivalently removes the parallel branches in Dialted Reparam Blocks, Batch Norm layers, and the bias term in GRN. The pseudo-code of the full pipeline will be like
    ```python
    training_model = unireplknet_l(...,  deploy=False)
    train(training_model)
    trained_results = evaluate(training_model)
    training_model.reparameterize_unireplknet()
    inference_results = evaluate(training_model)
    # you will see inference_results are identical to trained_results
    save(training_model, 'converted_weights.pth')
    # use the converted model
    deploy_model = unireplknet_l(..., deploy=True)
    load_weights(deploy_model, 'converted_weights.pth')
    deploy_results = evaluate(deploy_model)
    # you will see deploy_results == inference_results == trained_results
    ```
4. You may want to read this if you are familiar with the [timm](https://github.com/huggingface/pytorch-image-models/tree/main) library. We sincerely thank timm for providing a convenient [re-parameterize function](https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model.py#L225). The code design of UniRepLKNet is compatible with it. That is, calling ```some_unireplknet_model.reparameterize_unireplknet()``` is equivalent to calling ```timm.utils.reparameterize_model(some_unireplknet_model)```. So if you use our code with timm's codebase, e.g., timm's evaluation code, just add ```--reparam``` to your command so that ```timm.utils.reparameterize_model``` will be called (see [here](https://github.com/huggingface/pytorch-image-models/blob/main/validate.py#L128)).



## Models

We have provided five ways to download our checkpoints.

1. Download via the Google Drive links shown below.
2. Visit our huggingface repo at https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main and click the download icons.
3. Use huggingface-hub in your python code. First, install huggingface_hub
```
pip install huggingface_hub
```
Then, use huggingface_hub like this in your python code, for example,
```python
from huggingface_hub import hf_hub_download
repo_id = 'DingXiaoH/UniRepLKNet'
cache_file = hf_hub_download(repo_id=repo_id, filename=FILE_NAME)
checkpoint = torch.load(cache_file, map_location='cpu')
model.load_state_dict(checkpoint)
```
See our [huggingface repo](https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main) or [our code](unireplknet.py#L670) for FILE_NAME (e.g., ```unireplknet_xl_in22k_pretrain.pth```).

4. Use the huggingface CLI. Check the [tutorial](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

5. Automatically download our checkpoints by passing ```in_1k_pretrained=True```, ```in_22k_pretrained=True```, or ```in_22k_to_1k=False``` while calling our provided functions. See the [code here](unireplknet.py#L738).

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
| UniRepLKNet-S | 224x224 | 56M | 26.7G  | (lost. reproducing) |
| UniRepLKNet-B | 224x224 | 98M   | 47.2G   | [ckpt](https://drive.google.com/file/d/1t1txZOTpwXGUMVsqyUxpzE5EGLqMX5li/view?usp=drive_link)|
| UniRepLKNet-L | 192x192 | 218M  | 105.4G   | [ckpt](https://drive.google.com/file/d/1PEY474n6a7pZ3vJitsU7ZLzwBI00pf7u/view?usp=drive_link)|
| UniRepLKNet-XL | 192x192 | 386M  | 187G  | [ckpt](https://drive.google.com/file/d/1OP7I0jabljm8LKXTypk4HDmF9dQQqYib/view?usp=drive_link)|

### Pretrained on ImageNet-22K then finetuned on ImageNet-1K

| name | resolution |acc@1 | #params | FLOPs | ckpt |
|:---:|:---:|:---:|:---:| :---:| :---:|
| UniRepLKNet-S | 384x384 | 86.4 | 56M | 26.7G  | [ckpt](https://drive.google.com/file/d/1PzEHFOgEllMRIB-emkX_2VjXyBYC_X0z/view?usp=drive_link)|
| UniRepLKNet-B | 384x384 | 87.4 | 98M   | 47.2G   | [ckpt](https://drive.google.com/file/d/1T4BB3xx6FsWrK5QpTy7FwBrLuOMcZcEu/view?usp=drive_link)|
| UniRepLKNet-L | 384x384 | 87.9 | 218M  | 105.4G   | [ckpt](https://drive.google.com/file/d/10jJGzXX3cFRrfk3oAoIoWRnKSAaquQtM/view?usp=drive_link)|
| UniRepLKNet-XL | 384x384 | 88.0 | 386M  | 187G  | (lost. reproducing)|

### COCO Object Detection

Code, document, and config files have been released. See the [detection guide](detection/README.md) here.

Checkpoints have already been released on hugging face. Please see https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main. You can download them right now.

Or you can download these checkpoints with Google Drive as follows:


| name | resolution |box mAP | mask mAP | #params | FLOPs | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| UniRepLKNet-T | 1280x800 | 51.7 | 44.9 | 89M  | 749G | [ckpt](https://drive.google.com/file/d/15LVEXyC8xxOIHhUeSFeolyZ1IVXVjQ4I/view?usp=drive_link) |
| UniRepLKNet-S | 1280x800 | 53.0 | 45.9 | 113M  | 835G | [ckpt](https://drive.google.com/file/d/1wcdMn35aMLgjIFVEIaJYMjOwMtuoz58I/view?usp=drive_link) |
| UniRepLKNet-S_22K | 1280x800 | 54.3 | 47.1 | 113M  | 835G | [ckpt](https://drive.google.com/file/d/1pZmrLRbM8bjiQvr_xenReXmtA7M3f_Ii/view?usp=sharing) |
| UniRepLKNet-B_22K | 1280x800 | 54.8 | 47.4 | 155M  | 978G | [ckpt](https://drive.google.com/file/d/1CCyk0q4E4tuFLWqafHIC-DywdJJ0-pMQ/view?usp=drive_link) |
| UniRepLKNet-L_22K | 1280x800 | 55.8 | 48.4 | 276M  | 1385G | [ckpt](https://drive.google.com/file/d/1m9WzhfhEF1KKxLH8IxE5vkM7HlucJu4N/view?usp=drive_link) |
| UniRepLKNet-XL_22K | 1280x800 | 56.4 | 49.0 | 443M  | 1952G | [ckpt](https://drive.google.com/file/d/1np1zCV_34MdOsViKMVdO1l4feLe8evmp/view?usp=drive_link) |

### ADE-20K Semantic Segmentation

Code, document, and config files have been released. See the [segmentation guide](segmentation/README.md) here.

Checkpoints have already been released on hugging face. Please see https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main. You can download them right now.

Or you can download these checkpoints with Google Drive as follows:

| name | resolution |mIoU (ss/ms) | #params | FLOPs | Weights |
|:---:|:---:|:---:|:---:| :---:|:---:|
| UniRepLKNet-T | 512x512 | 48.6/49.1 | 61M | 946G  | [ckpt](https://drive.google.com/file/d/1R2teeQt7q48EBBRbeVXShISpOmS5YHjs/view?usp=drive_link) |
| UniRepLKNet-S | 512x512 | 50.5/51.0 | 86M  | 1036G | [ckpt](https://drive.google.com/file/d/1SBHvbK4zoPSZ827F5Sp209LYIh2T7Iew/view?usp=drive_link) |
| UniRepLKNet-S_22K | 512x512 | 51.9/52.7 | 86M  | 1036G | [ckpt](https://drive.google.com/file/d/15dNuw34kia5qtt6UijcnutEktY05OrKH/view?usp=drive_link) |
| UniRepLKNet-S_22K | 640x640 | TBA | 86M  | TBA |  TBA |
| UniRepLKNet-B_22K | 640x640 | 53.5/53.9 | 130M  | 1850G | [ckpt](https://drive.google.com/file/d/1sflCn8ny-cU5Bk8yBGE3E-yIO8eECE0H/view?usp=drive_link) |
| UniRepLKNet-L_22K | 640x640 | 54.5/55.0 | 254M  | 2507G | [ckpt](https://drive.google.com/file/d/1Qev75aKZY5bNAM17cLecD2OoZwKf5DA7/view?usp=drive_link) |
| UniRepLKNet-XL_22K | 640x640 | 55.2/55.6 | 425M  | 3420G | TBA |

## ImageNet evaluation and training

We give an example evaluation command.

Single-GPU
```
python main.py --model unireplknet_b --eval true \
--resume unireplknet_b_in22k_to_in1k_384_acc87.40.pth  \
--input_size 384 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model unireplknet_b --eval true \
--resume unireplknet_b_in22k_to_in1k_384_acc87.40.pth  \
--input_size 384 \
--data_path /path/to/imagenet-1k
```
For training or finetuning UniRepLKNets on ImageNet-1K or 22K, see [THIS DOC](/Image/README.md)

## Universal perception of audio, video, point cloud, and time-series tasks

For detailed documentation, please refer to these documents as follows:

* [Audio DOC](/Audio/README.md)
* [Point Cloud DOC](/Point/README.md)
* [Time-Series DOC](/Time-Series/README.md)
* [Video DOC](/Video/README.md)

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
