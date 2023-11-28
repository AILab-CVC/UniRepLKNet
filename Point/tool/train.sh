#!/bin/sh
# LOG
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
#set -o pipefail

exp_name=$1
config=$2
dataset=$3

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2
export KMP_INIT_AT_FORK=FALSE

PYTHON=python
TRAIN_CODE=train.py
TEST_CODE=test.py


exp_dir=Exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result

now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

export PYTHONPATH=.
#rm -rf /dev/shm/wbhu*
echo $OMP_NUM_THREADS | tee -a ${exp_dir}/train-$now.log
nvidia-smi | tee -a ${exp_dir}/train-$now.log
which pip | tee -a ${exp_dir}/train-$now.log


$PYTHON -u ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee -a ${exp_dir}/train-$now.log

# TEST
#rm -rf /dev/shm/wbhu*
now=$(date +"%Y%m%d_%H%M%S")

$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee -a ${exp_dir}/test_best-$now.log

$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/last \
  model_path ${model_dir}/model_last.pth \
  2>&1 | tee -a ${exp_dir}/test_last-$now.log
