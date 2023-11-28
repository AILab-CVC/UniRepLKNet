#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/path/to/your/log'
DATA_PATH='../data/kinetics400'
# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    --master_port 12319 run_class_finetuning.py \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --update_freq 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --sampling_rate 4 \
    --update_freq 2 \
    --opt adamw \
    --lr 4e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 90 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \