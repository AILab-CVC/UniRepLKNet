# UniRepLKNet for Object Detection

This folder contains the implementation of UniRepLKNet for object detection.

Our detection code is developed on top of [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/tree/v2.28.1).

### Install

- Clone this repo:

```bash
git clone https://github.com/AILab-CVC/UniRepLKNet.git
cd UniRepLKNet
```

- Create a conda virtual environment and activate it:

```bash
conda create -n unireplknet python=3.7 -y
conda activate unireplknet
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.1](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).


### Evaluation

To evaluate UniRepLKNets on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm
```

For example, to evaluate the `UniRepLKNet-XL` with a single GPU:

```bash
python test.py configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_internimage_t_fpn_1x_coco.pth --eval bbox segm
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/mask_rcnn_internimage_b_fpn_1x_coco.py checkpoint_dir/det/mask_rcnn_internimage_b_fpn_1x_coco.py 8 --eval bbox segm
```

### Training on COCO

To train a model on COCO, run

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node, run:

```bash
sh dist_train.sh configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py 8
```


### Export

To export a detection model from PyTorch to TensorRT, run:
```shell
MODEL="config_file_name"
CKPT_PATH="/path/to/model/ckpt.pth"

python deploy.py \
    "./deploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py" \
    "./configs/coco/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.jpg" \
    --work-dir "./work_dirs/mmdet/instance-seg/${MODEL}" \
    --device cuda \
    --dump-info
```

For example, to export `mask_rcnn_internimage_t_fpn_1x_coco` from PyTorch to TensorRT, run:
```shell
MODEL="mask_rcnn_internimage_t_fpn_1x_coco"
CKPT_PATH="/path/to/model/ckpt/mask_rcnn_internimage_t_fpn_1x_coco.pth"

python deploy.py \
    "./deploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py" \
    "./configs/coco/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.jpg" \
    --work-dir "./work_dirs/mmdet/instance-seg/${MODEL}" \
    --device cuda \
    --dump-info
```

### Acknowledgements 

We directly used the detection codebase of InternImage for a strict fair comparison, and I would like to express my sincere gratitude to the InternImage project for their comprehensive and well-written documentation. This README has been adapted from their object detection README file. All credit for the structure and clarity of this README goes to the original authors of InternImage.