# Audio Spectrogram Understanding with UniRepLKNet

Created by [Xiaohan Ding](http://dingxiaohan.xyz/), [Yiyuan Zhang](https://scholar.google.com/citations?hl=en&user=KuYlJCIAAAAJ), etc.


This repository is an official implementation of **UniRepLKNet** .


This repository is built to explore the ability of RepLK-series networks to understand audio spectrograms. Following common practice in [AST](https://github.com/YuanGongND/ast), we also transform raw waves into Mel features with a spatial-relevant form. Then we employ large-kernel convnets to deal with speech classification.


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
```
- *( Note that the python virtual environment of audio understanding is compatible with [point cloud understanding](../Point/README.md))*
### Data Preparation

- The data of speech commands v2 can be directly downloaded:
  ```
  cd egs/speechcommands && bash run_sc.sh
  ```

- The data is expected to be in the following file structure:
    ```
    Audio/
    |-- src/
    |-- egs/
        |-- Speechcommands/
            |-- data/
                |-- datafiles/
                    | -- speechcommand_eval_data.json
                    | -- speechcommand_train_data.json
                    | -- speechcommand_valid_data.json
                |-- speech_commands_v0.02/
                |-- speechcommands_class_labels_indices.csv
    |-- pretrained_models/
    ```

- *( Other datasets can be found in [AST](https://github.com/YuanGongND/ast))*

## Usage

### Train

```
bash run_sc.sh
```
- *( Please according to your practical settings to modify these Variables)*

## Acknowledgements

Our code is based on [AST](https://github.com/YuanGongND/ast).
