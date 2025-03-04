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
export MASTER_PORT=12345
export WORLD_SIZE=2
echo "Using Master_Addr $MASTER_ADDR:$MASTER_PORT to synchronise, and a world size of $WORLD_SIZE."
#
#
echo "Setting path to dataset (ImageNet), as well as output directory for checkpoints"
export DATAPATH='/mnt/datasets/ILSVRC2012'
export MODEL_CHECKPOINT='/mnt/<your_path_to_the_model_checkpoint>/model_best.pth.tar'
#
#
echo "Start evaluating the default tiny BiXT model with 64 latents, using 2GPUs and a total validation batch size of 4096..."
torchrun --nproc_per_node=2 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT evaluate_BiXT.py --model bixt_ti_l64_p16 --model_checkpoint $MODEL_CHECKPOINT --workers 6 --data_path $DATAPATH --validation_batch_size 2048
echo "Finished evaluating. Cheers!"

