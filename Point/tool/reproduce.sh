#!/bin/sh
# LOG
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
#set -o pipefail

DATASET=$1
MODEL=$2

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2
export KMP_INIT_AT_FORK=FALSE
export PYTHONPATH=.

exp_dir=pretrained/reproduce
result_dir=${exp_dir}/result
ckpt_dir=${exp_dir}/ckpt
log_dir=${exp_dir}/log

mkdir -p ${exp_dir} ${result_dir} ${log_dir}
mkdir -p ${log_dir}/${DATASET}

# TEST

python -u tool/test.py --config=config/${DATASET}/p2p_${MODEL}.yaml \
  save_folder ${result_dir} \
  model_path ${ckpt_dir}/${DATASET}/${MODEL}-${DATASET}.pth \
  2>&1 | tee -a ${log_dir}/${DATASET}/test_${MODEL}-${DATASET}.log

