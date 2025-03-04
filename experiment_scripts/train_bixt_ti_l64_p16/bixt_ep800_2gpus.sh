#!/bin/bash
#
#echo "Starting up environment..."
#source /<path_to_miniconda3>/bin/activate bixt_env
#echo "... done!"
#
cd ../..
#
#
export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES='0,1'
#
export WANDB_API_KEY="<INSERT_YOUR_KEY_HERE>"  # and activate using argument, if tracking via wandb is desired
#
echo "Attempting to set up training..."
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12346
export WORLD_SIZE=2
echo "Using Master_Addr $MASTER_ADDR:$MASTER_PORT to synchronise, and a world size of $WORLD_SIZE."
#
#
echo "Setting path to dataset (ImageNet), as well as output directory for checkpoints"
export DATAPATH='/mnt/datasets/ILSVRC2012'
export OUTPUT_DIR='/mnt/<your_checkpoint_path>/BiXT_checkpoints'
#
echo "Start training the default tiny BiXT model with 64 latents, using the default hyperparameters and total batch size of 1024..."
torchrun --nproc_per_node=2 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT train_BiXT.py --torchcompile inductor --model bixt_ti_l64_p16 --seed 42 --lr 2.5e-3 --sa_drop_path 0.00 --ca_drop_path 0.00 --workers 6 --warmup_lr 1e-6 --data_path $DATAPATH --batch_size_per_gpu 512 --epochs 800 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 5 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
echo "Finished training. Cheers!"

