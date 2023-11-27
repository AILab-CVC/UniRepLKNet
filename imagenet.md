## ImageNet-1K Training

We use an initial learning rate of 4e-3 and a total batch size of 4096, which means num_nodes * num_gpus_per_node * batch_size * update_freq = 4096. You may increase the update_freq if you have insufficient GPUs or GPU memory.

If you desire to change the total batch size, linearly scaling the initial learning rate will result in comparable results (e.g., use an initial learning rate of 2e-3 if your total batch size is 2048).

If you get OOM (Out-Of-Memory) error, you may reduce the batch size or try ```--use_checkpoint true```, which uses torch.utils.checkpoint to significantly save GPU memory for training (it has nothing to do with the "checkpoint" referring to the weights saved in a file).

We used a single machine to train all the ImageNet-1K-only models.

UniRepLKNet-A was trained with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_a --drop_path 0.0 \
--batch_size 128 --lr 4e-3 --update_freq 8 \
--mixup 0.3 --cutmix 0.3 \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
UniRepLKNet-F was trained with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_f --drop_path 0.0 \
--batch_size 128 --lr 4e-3 --update_freq 8 \
--mixup 0.3 --cutmix 0.3 \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```
UniRepLKNet-P was trained with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_p --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 8 \
--mixup 0.3 --cutmix 0.3 \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```
UniRepLKNet-N was trained with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_n --drop_path 0.1 \
--batch_size 128 --lr 4e-3 --update_freq 8 \
--mixup 0.5 --cutmix 0.5 \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```
UniRepLKNet-T was trained with 4 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_t --drop_path 0.2 \
--batch_size 128 --lr 4e-3 --update_freq 8 \
--mixup 0.8 --cutmix 1.0 \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```
UniRepLKNet-S was trained with 8 GPUs.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model unireplknet_s --drop_path 0.4 \
--batch_size 64 --lr 4e-3 --update_freq 8 \
--mixup 0.8 --cutmix 1.0 \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```

## ImageNet-22K Pretraining

We used 2 machines each with 8 GPUs to train the S/B/L models and 4 machines to train XL on ImageNet-22K.

You may use multi-node training on a SLURM cluster with [submitit](https://github.com/facebookincubator/submitit). Please install:
```
pip install submitit
```

You may alternatively use a single machine and larger update_freq or larger batch size, or a smaller total batch size with a smaller initial learning rate.

UniRepLKNet-S used similar configurations to ConvNeXt for fair comparison with ConvNeXt-S (we used num_nodes=2 and update_freq=8 while ConvNeXt used num_nodes=16 and update_freq=1).
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model unireplknet_s --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 8 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true 
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```
UniRepLKNet-B used similar configurations to ConvNeXt for fair comparison with ConvNeXt-B.
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model unireplknet_b --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 8 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true 
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```
UniRepLKNet-L used similar configurations (initial learning rate of 1e-3 and input size of 192) to InternImage for fair comparison with InternImage-L.
```
python run_with_submitit.py --nodes 2 --ngpus 8 \
--model unireplknet_l --drop_path 0.1 \
--batch_size 64 --lr 1e-3 --update_freq 4 \
--input_size 192 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true 
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```
UniRepLKNet-XL used similar configurations to InternImage for fair comparison with InternImage-XL.
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model unireplknet_xl --drop_path 0.2 \
--batch_size 32 --lr 1e-3 --update_freq 4 \
--input_size 192 \
--warmup_epochs 5 --epochs 90 \
--data_set image_folder --nb_classes 21841 --disable_eval true 
--data_path /path/to/imagenet-1k \
--job_dir /path/to/save_results
```

## Finetune ImageNet-22K-pretrained models with ImageNet-1K

While finetuning the ImageNet-22K-pretrained models with ImageNet-1K, we used a total batch size of 512 (num_nodes * num_gpus_per_node * batch_size * update_freq = 512). We used weight decay of 1e-8 and no mixup/cutmix.

UniRepLKNet-S used similar configurations to ConvNeXt for fair comparison with ConvNeXt-S.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_s --drop_path 0.2 --input_size 384 \
--batch_size 64 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
UniRepLKNet-B used similar configurations to ConvNeXt for fair comparison with ConvNeXt-B.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_b --drop_path 0.2 --input_size 384 \
--batch_size 64 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
UniRepLKNet-L used similar configurations (20 epochs, label smoothing of 0.3, and usage of EMA) to InternImage for fair comparison with InternImage-L.
```
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model unireplknet_l --drop_path 0.3 --input_size 384 \
--batch_size 64 --lr 5e-5 --update_freq 2 \
--model_ema true --model_ema_eval true \
--warmup_epochs 0 --epochs 20 --weight_decay 1e-8 --smoothing 0.3 \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
UniRepLKNet-XL used similar configurations to InternImage for fair comparison with InternImage-XL.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model unireplknet_xl --drop_path 0.3 --input_size 384 \
--batch_size 32 --lr 5e-5 --update_freq 2 \
--model_ema true --model_ema_eval true \
--warmup_epochs 0 --epochs 20 --weight_decay 1e-8 --smoothing 0.3 \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```

## Acknowledgement
The released PyTorch ImageNet training script is based on the code of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), which was built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories.
