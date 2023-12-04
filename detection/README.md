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

You can download checkpoint files from our Google Drive or Hugging Face repo.

To evaluate UniRepLKNets on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm --cfg-options model.backbone.init_cfg.checkpoint=None
```
Note that we use ```--cfg-options model.backbone.init_cfg.checkpoint=None``` to overwrite the initialization config of the backbone so that its initialization with the ImageNet-pretrained weights will be skipped. This is because we will load its weights together with the Cascade Mask RCNN heads from the checkpoint file.

You may also 1) change ```checkpoint``` to ```None``` in the config file to realize the same effect or 2) simply ignore it if you have downloaded the ImageNet-pretrained weights (initializing the backbone twice does no harm except for wasting time).

For example, to evaluate the `UniRepLKNet-T` with a single GPU:

```bash
python test.py configs/coco/casc_mask_rcnn_unireplknet_t_in1k_fpn_3x_coco.py casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75.pth --eval bbox segm
```

For example, to evaluate the `UniRepLKNet-XL` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/coco/casc_mask_rcnn_unireplknet_xl_in22k_fpn_3x_coco.py casc_mask_rcnn_unireplknet_xl_in22k_3x_coco_ap56.39.pth 8 --eval bbox segm
```

### Training on COCO

To train a `UniRepLKNet` on COCO, 1) ensure that the ```init_cfg.checkpoint``` in the config file refers to the downloaded pretrained weights, and 2) run

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `UniRepLKNet-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/coco/casc_mask_rcnn_unireplknet_t_in1k_fpn_3x_coco.py 8
```



### Manage Jobs with Slurm

For example, to train `UniRepLKNet-B` with 32 GPU on 4 node, run:

```bash
GPUS=32 sh slurm_train.sh <partition> <job-name> configs/coco/casc_mask_rcnn_unireplknet_b_in22k_fpn_3x_coco.py your_work_dir

```


### Re-parameterize Trained Model into the Inference-time Structure

Before deployment, we may equivalently remove the following structures to convert the trained model into the inference structure for deployment: parallel branches in Dilated Reparam Block, BatchNorms, and the bias term in GRN.

The following command calls ```UniRepLKNet.reparameterize_unireplknet()``` and the resultant weights will be saved to ```casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75_deploy.pth```.

```
python3 reparameterize.py configs/coco/casc_mask_rcnn_unireplknet_t_in1k_fpn_3x_coco.py UniRepLKNet/casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75.pth casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75_deploy.pth --cfg-options model.backbone.init_cfg.checkpoint=None
```

Then test the converted model. Note we use ```model.backbone.deploy=True``` to overwrite the original configuration in the config file so that the model will be constructed in the inference form (i.e., without the parallel branches, BatchNorms, and bias in GRN).
```
python3 test.py configs/coco/casc_mask_rcnn_unireplknet_t_in1k_fpn_3x_coco.py casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75_deploy.pth --cfg-options model.backbone.deploy=True, model.backbone.init_cfg.checkpoint=None --eval mIoU
```

You will get identical results to the trained model.



### Export

To export a detection model from PyTorch to TensorRT, for example, run:
```
python deploy.py \
    "./deploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py" \
    "./configs/coco/casc_mask_rcnn_unireplknet_t_in1k_fpn_3x_coco_deploy.py" \
    "casc_mask_rcnn_unireplknet_t_in1k_3x_coco_ap51.75_deploy.pth" \
    "./deploy/demo.jpg" \
    --work-dir "./work_dirs/mmdet/instance-seg/${MODEL}" \
    --device cuda \
    --dump-info
```


### Acknowledgements 

We directly used the detection codebase of InternImage for a strict fair comparison, and I would like to express my sincere gratitude to the InternImage project for their comprehensive and well-written documentation. This README has been adapted from their object detection README file. All credit for the structure and clarity of this README goes to the original authors of InternImage.