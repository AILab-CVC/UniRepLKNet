# Video Recognition with UniRepLKNet

In this part, we explore utilizing a single convnet for video recognition task. Without special design, we just simply reshape the input tensors, and UniRepLKNet can deliver comparable performances better than other generlalist models. 
This part of code is based on [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2). Thanks for their outstanding project.
## Usage

### 1. Environment Setup.

```
pip install -r requirements.txt
```


### 2. Train and evaluate model. 
We provide the experiment scripts for easier use:

* Kinetics-400 Dataset
```
bash run_unireplknet.sh
```
*Please edit `run_unireplknet.sh` before running the code.*
