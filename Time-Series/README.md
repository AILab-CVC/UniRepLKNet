# Time-Series Forecastin with UniRepLKNet

We explore utilize advanced visual architectures for time-series forecasting tasks. We conduct experiments on the global temperature and wind speed forecasting. Our UniRepLKNet achieves new state-of-the-art performances on both challenging tasks.

This repository is built on [Corrformer](https://www.nature.com/articles/s42256-023-00667-9), we appreciate their great contributions.


## Code Structure

```python
|-- Time-Series
   |-- data_provider # Data loader
   |-- exp # Pipelines for train, validation and test
   |-- layers 
   |   |-- Embed.py # Temporal Embeddings
   |   |-- Causal_Conv.py # Causal conv for Cross-Correlation
   |   |-- Multi_Correlation.py # Equ (5)-(10) of the paper
   |-- models
   |   |-- Corrformer.py # The Framework to employ UniRepLKNet
   |-- utils
   |-- scripts # Running scripts
   |-- dataset # Place the download datsets here
   |-- checkpoints # Place the output or pretrained models here
```

## Reproduction

1. Find a device with GPU support. Our experiment is conducted on a single RTX 24GB GPU and in the Linux system.
2. Install Python 3.6, PyTorch 1.7.1. The following script can be convenient.

```bash
pip install -r requirements.txt # take about 5 minutes
```

2. Download the dataset from [[Code Ocean]](https://codeocean.com/capsule/0341365/tree/v1). And place them under the `./dataset` folder.

3. Train and evaluate the model with the following scripts.

```shell
bash ./scripts/Global_Temp/UniRepLKNet.sh # take about 18 hours
bash ./scripts/Global_Wind/UniRepLKNet.sh # take about 18 hours
```

Note: Since the raw data for Global Temp and Global Wind from the NCEI has been multiplied by ten times, the actual MSE and MAE for these two benchmarks should be divided by 100 and 10 respectively.