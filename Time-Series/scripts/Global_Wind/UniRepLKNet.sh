export CUDA_VISIBLE_DEVICES=2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/global_wind/ \
  --data_path None \
  --model_id UniRepLKNet_global_wind_48_24_1ETCN_1DTCN \
  --model Corrformer \
  --data Global_Wind \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --factor_temporal 1 \
  --factor_spatial 1 \
  --enc_tcn_layers 1 \
  --dec_tcn_layers 1 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --node_num 350 \
  --node_list 7,50 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --batch_size 1 \
  --n_heads 16