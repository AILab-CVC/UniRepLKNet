# UniRepLKNet for Semantic Segmentation

This folder contains the implementation of the UniRepLKNet for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## Usage

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

For examples, to install torch==1.11 with CUDA==11.3 and nvcc:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge cudatoolkit-dev=11.3 -y # to install nvcc
```

- Install other requirements:

  note: conda opencv will break torchvision as not to support GPU, so we need to install opencv using pip. 	  

```bash
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
```

- Install `timm` and `mmcv-full` and `mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```


### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `UniRepLKNet` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```
You can download checkpoint files from our Google Drive or Hugging Face repo. Then place it to segmentation/checkpoint_dir/seg.

For example, to evaluate the `UniRepLKNet-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_unireplknet_t_512_160k_ade20k.py checkpoint_dir/seg/upernet_unireplknet_t_512_160k_ade20k_miou48.56.pth --eval mIoU
```

For example, to evaluate the `UniRepLKNet-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_unireplknet_b_in22k_640_160k_ade20k.py checkpoint_dir/seg/upernet_unireplknet_b_in22k_640_160k_ade20k_miou53.52.pth 8 --eval mIoU
```

### Training

To train a `UniRepLKNet` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `UniRepLKNet-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_unireplknet_t_512_160k_ade20k.py 8
```

### Manage Jobs with Slurm

For example, to train `UniRepLKNet-B` with 8 GPU on 1 node (total batch size 16), run:

```bash
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/ade20k/upernet_unireplknet_b_in22k_640_160k_ade20k.py
```

### Image Demo
To inference a single/multiple image like this.
If you specify image containing directory instead of a single image, it will process all the images in the directory.:
```
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/ade20k/upernet_unireplknet_t_512_160k_ade20k.py  \
  checkpoint_dir/seg/upernet_unireplknet_t_512_160k_ade20k_miou48.pth  \
  --palette ade20k 
```

### Re-parameterize Trained Model into the Inference-time Structure

Equivalently remove the following structures to convert the trained model into the inference structure for deployment: parallel branches in Dilated Reparam Block, BatchNorms, and the bias term in GRN.

The following command calls ```UniRepLKNet.reparameterize_unireplknet()``` and the resultant weights will be saved to ```upernet_unireplknet_t_512_160k_ade20k_miou48.56_deploy.pth```.

Note that we use ```model.backbone.init_cfg.checkpoint=None``` to overwrite the initialization config of the backbone because we will load the trained segmentation model ```upernet_unireplknet_t_512_160k_ade20k_miou48.56.pth```. 

```
python3 reparameterize.py configs/ade20k/upernet_unireplknet_t_512_160k_ade20k.py upernet_unireplknet_t_512_160k_ade20k_miou48.56.pth upernet_unireplknet_t_512_160k_ade20k_miou48.56_deploy.pth --cfg-options model.backbone.init_cfg.checkpoint=None
```
Then test the converted model. Note we use ```model.backbone.deploy=True``` to overwrite the original configuration in the config file so that the model will be constructed in the inference form (i.e., without the parallel branches, BatchNorms, and bias in GRN).
```
python3 test.py configs/ade20k/upernet_unireplknet_t_512_160k_ade20k.py upernet_unireplknet_t_512_160k_ade20k_miou48.56_deploy.pth --cfg-options model.backbone.deploy=True, model.backbone.init_cfg.checkpoint=None --eval mIoU
```

### Export

To export a segmentation model from PyTorch to TensorRT, for example, run:
```
python deploy.py \
    "./deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "./configs/ade20k/upernet_unireplknet_t_512_160k_ade20k_deploy.py" \
    "upernet_unireplknet_t_512_160k_ade20k_miou48.56_deploy.pth" \
    "./deploy/demo.png" \
    --work-dir "./work_dirs/mmseg/upernet_unireplknet_t" \
    --device cuda \
    --dump-info
```


### Acknowledgements 

We directly used the segmentation codebase of InternImage for a strict fair comparison, and I would like to express my sincere gratitude to the InternImage project for their comprehensive and well-written documentation. This README has been adapted from their semantic segmentation README file. All credit for the structure and clarity of this README goes to the original authors of InternImage.