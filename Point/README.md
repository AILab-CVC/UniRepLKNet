# Point Cloud Understanding with UniRepLKNet

Created by [Xiaohan Ding](http://dingxiaohan.xyz/), [Yiyuan Zhang](https://scholar.google.com/citations?hl=en&user=KuYlJCIAAAAJ), etc.


This repository is an official implementation of **UniRepLKNet**.


This repository is built to explore the ability of RepLK-series networks to understand point cloud. We are mainly focused on the shape classification with ModelNet-40 and ScanObjectNN datasets. Besides fully training, we also explore the advantages of pretrained UniRepLKNet on few-shot learning tasks.

## Preparation

### Installation Prerequisites

- Python 3.9
- CUDA 11.3
- PyTorch 1.11.1
- timm 0.5.4
- torch_scatter
- pointnet2_ops
- cv2, sklearn, yaml, h5py

```
conda create -n pt python=3.9
conda activate pt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

mkdir lib
cd lib
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install pointnet2_ops_lib/.
cd ../..

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install timm==0.5.4 opencv-python scikit-learn h5py pyyaml tqdm tensorboardx einops
```

### Data Preparation

- Download the processed ModelNet40 dataset from [[Google Drive]](https://drive.google.com/drive/folders/1fAx8Jquh5ES92g1zm2WG6_ozgkwgHhUq?usp=sharing)[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/)[[BaiDuYun]](https://pan.baidu.com/s/18XL4_HWMlAS_5DUH-T6CjA )(code:4u1e). Or you can download the offical ModelNet from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and process it by yourself.

- Download the official ScanObjectNN dataset from [here](http://103.24.77.34/scanobjectnn).

- The data is expected to be in the following file structure:
    ```
    Point/
    |-- config/
    |-- data/
        |-- ModelNet40/
            |-- modelnet40_shape_names.txt
            |-- modelnet_train.txt
            |-- modelnet_test.txt
            |-- modelnet40_train_8192pts_fps.dat
            |-- modelnet40_test_8192pts_fps.dat
        |-- ScanObjectNN/
            |-- main_split/
                |-- training_objectdataset_augmentedrot_scale75.h5
                |-- test_objectdataset_augmentedrot_scale75.h5
    |-- dataset/
    ```

*(modelnet40_shape_names.txt, modelnet_train.txt, and modelnet_test.txt are provided in [PointBERT](https://github.com/lulutang0608/Point-BERT/tree/master/data/ModelNet/modelnet40_normal_resampled) )*

## Usage

```
bash tool/train_unireplknet.sh mv_unireplket-s ModelNet40 config/ModelNet40/multiview_UniRepLKNet-S.yaml
```


## Acknowledgements

Our code is inspired by [Meta-Transformer](https://arxiv.org/abs/2307.10802) and [P2P](http://arxiv.org/abs/2208.02812).
