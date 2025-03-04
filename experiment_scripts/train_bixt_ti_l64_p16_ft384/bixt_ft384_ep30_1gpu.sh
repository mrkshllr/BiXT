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
export CUDA_VISIBLE_DEVICES='0'
#
export WANDB_API_KEY="<INSERT_YOUR_KEY_HERE>"  # and activate using argument, if tracking via wandb is desired
#
echo "Attempting to set up training..."
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export WORLD_SIZE=1
echo "Using Master_Addr $MASTER_ADDR:$MASTER_PORT to synchronise, and a world size of $WORLD_SIZE."
#
#
echo "Setting path to dataset (ImageNet), as well as output directory for checkpoints"
export DATAPATH='/mnt/datasets/ILSVRC2012'
export OUTPUT_DIR='/mnt/<your_checkpoint_path>/BiXT_checkpoints/FT'
#
#echo "Start fine-tuning on 384x384 images, using total batch size of 512..."
#torchrun --nproc_per_node=1 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT main.py --model bixt_ti_l64_p16_ft384_e800 --input_size 3 384 384 --pretrained --seed 42 --lr 2.5e-5 --sa_drop_path 0.05 --ca_drop_path 0.05 --workers 6 --torchcompile inductor --data_path $DATAPATH --batch_size_per_gpu 512 --epochs 30 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 0 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
#echo "Finished training. Cheers!"

## OR SIMPLY USE THE FOLLOWING:
echo "Start fine-tuning on 384x384 images, using total batch size of 512..."
python3 train_BiXT.py --model bixt_ti_l64_p16_ft384_e800 --input_size 3 384 384 --pretrained --seed 42 --lr 2.5e-5 --sa_drop_path 0.05 --ca_drop_path 0.05 --workers 6 --torchcompile inductor --data_path $DATAPATH --batch_size_per_gpu 512 --epochs 30 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 0 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
echo "Finished training. Cheers!"
