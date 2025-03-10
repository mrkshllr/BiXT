# BiXT - Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers 
Official PyTorch implementation of the paper **Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers** (NeurIPS 2024).
<div align="center">

:mortar_board: :page_facing_up: Find our paper: [[arXiv]](https://arxiv.org/pdf/2402.12138) &nbsp; [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ab1ee157f7804a13f980414b644a9460-Abstract-Conference.html) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; :milky_way: :chart_with_upwards_trend: [[Poster]](.github/Hiller_Poster_BiXT_NeurIPS2024_s.png) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; :mailbox_with_mail: Reference: [[BibTeX]](https://github.com/mrkshllr/BiXT#citing-bixt)
</div>

## Note :pencil:
- :arrow_right: **BiXT architecture** implemented [here!](timm/models/bixt.py) :bookmark: :point_left:
- :arrow_right: **Trained models** available [here!](https://github.com/mrkshllr/BiXT/blob/main/MODEL_WEIGHTS.md) :point_left: :computer:
- :arrow_right: Alternative BiXT architecture with conv-tokeniser implemented [here!](timm/models/bixt_convtok.py) :bookmark: :point_left:

## Updates :tada:
- March 10, 2025: **BiXT model weights** available; (pre-) trained on ImageNet1K, including d13-backbones for dense downstream tasks
- March 06, 2025: Extended Readme: Details on Training, Finetuning and Evaluation procedures :books: :nerd_face: 
- March 04, 2025: *Cleaned-up* **Model, Training and Evaluation code now available** (for ImageNet) :star2: :computer:
- December 13, 2024: **BiXT presented at NeurIPS 2024 in Vancouver, Canada** :mount_fuji: :snowflake:
- September 26, 2024: **BiXT is accepted at NeurIPS 2024!** :fire: 

## TL;DR :eyes:
<div align="center">
  <img width="95%" alt="BiXT concept" src=".github/BiXT_visual.png">
</div>

**BiXT** is a novel bi-directional Transformer architecture which scales linearly with input size in terms of computational cost and memory consumption, but does not suffer the drop in performance or limitation to only one input modality seen with other efficient Transformer-based approaches. BiXT is inspired by the Perceiver architectures but replaces iterative attention with an efficient bi-directional cross-attention module in which input tokens and latent variables attend to each other simultaneously, leveraging a naturally emerging attention-symmetry between the two. This approach unlocks a key bottleneck experienced by Perceiver-like architectures and enables the processing and interpretation of both semantics ('what') and location ('where') to develop alongside each other over multiple layers -- allowing its direct application to dense and instance-based tasks alike. By combining efficiency with the generality and performance of a full Transformer architecture, BiXT can process longer sequences like point clouds, text or images at higher feature resolutions and achieves competitive performance across a range of tasks like point cloud part segmentation, semantic image segmentation, image classification, hierarchical sequence modeling and document retrieval. Our experiments demonstrate that BiXT models outperform larger competitors by leveraging longer sequences more efficiently on vision tasks like classification and segmentation, and perform on par with full Transformer variants on sequence modeling and document retrieval -- but require 28% fewer FLOPs and are up to 8.4x faster.

---

## Installation and Datasets
For detailed instructions how to set up your environment, install required packages and get access to the ImageNet dataset, please refer to the [installation instructions](https://github.com/mrkshllr/BiXT/blob/main/INSTALL.md).


## Training BiXT from scratch
For a glimpse at the documentation of all arguments that can be adjusted for training, you can run the following command  
```
python3 train_BiXT.py --help
```
This will display all arguments that can be passed to the training file. 

> [!TIP]  
> :robot: :thought_balloon: For a list of all models that are currently implemented, take a look at the [BiXT model file](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L51) for our default versions, and at the [BiXT-convtok model file](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt_convtok.py#L51) for variants using a convolutional tokeniser (used mainly for ablations).

To make things easier, we provide a set of examples how to train BiXT models in the form of bash scripts in the [experiments_scripts](experiment_scripts) folder, together with a list of our default hyperparameter choices passed as arguments.

Please make sure to define the following environment variables accordingly (part of the provided scripts):

* `$DATAPATH`: path to your ImageNet dataset (e.g. /mnt/datasets/ILSVRC2012)
* `$OUTPUT_DIR`: path to directory where model checkpoints and other logging data shall be stored (e.g. /mnt/bixt_checkpoints). Each run creates a hash-subdirectory based on the provided arguments for the experiment to avoid unintended overwriting of data. 

For example, to train our BiXT-Ti/16 model with 64 latents on 224x224 images using 1 GPU, you can use [this script](experiment_scripts/train_bixt_ti_l64_p16/bixt_ep800_1gpu.sh), which internally calls
```
python3 train_BiXT.py --torchcompile inductor --model bixt_ti_l64_p16 --seed 42 --lr 2.5e-3 --sa_drop_path 0.00 --ca_drop_path 0.00 --workers 6 --warmup_lr 1e-6 --data_path $DATAPATH --batch_size_per_gpu 1024 --epochs 800 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 5 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
```

Distributing the training across multiple GPUs works in the same manner as shown in [this script](experiment_scripts/train_bixt_ti_l64_p16/bixt_ep800_2gpus.sh). For 2 GPUs, this results in calling
```
torchrun --nproc_per_node=2 --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT train_BiXT.py --torchcompile inductor --model bixt_ti_l64_p16 --seed 42 --lr 2.5e-3 --sa_drop_path 0.00 --ca_drop_path 0.00 --workers 6 --warmup_lr 1e-6 --data_path $DATAPATH --batch_size_per_gpu 512 --epochs 800 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 5 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
```
The `rdzv-endpoint` for the communication can be set by defining `$MASTER_ADDR` and `$MASTER_PORT` (not necessary if only running one experiment on the server at a time).

&nbsp;
> [!NOTE]  
> Note that we used a **total batch-size** of 1024 images for training all our BiXT models.  
> This must be adjusted accordingly for single/multi-GPU setups, as demonstrated in our scripts:  
> We use a `batch_size_per_gpu=1024` for training on 1GPU, a `batch_size_per_gpu=512` for training on 2GPUs, etc.   
> In case your GPU cannot fit the desired batch-size, you can use the `grad_accum_steps` to compute the gradients sequentially and aggregate, before backprop.

###Available weights of trained BiXT models
More insights regarding obtained ImageNet validation accuracies (top1 and top5), as well as the model weights are available [here](https://github.com/mrkshllr/BiXT/blob/main/MODEL_WEIGHTS.md#default-bixt-tiny-models).

&nbsp;

## Finetuning BiXT models
We use the same training script to finetune models trained on 224x224 images on the larger resolution 384x384.

For ease of use, we define separate models that are automatically initialised with the weights of the model pretrained on the smaller resolution, see [here](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L75) (e.g. `bixt_ti_t64_p16_ft384`).  
To automatically load the correct weights, the path to the respective model checkpoint needs to be added to the [model definition](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L75) as `file='<checkpoint_path>'` configuration argument:
```
'bixt_ti_l64_p16_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar')
```

Finetuning can then be started akin to training, either via the provided [finetuning scripts](experiment_scripts/train_bixt_ti_l64_p16_ft384) or by passing the appropriate arguments to 
```
python3 train_BiXT.py --model bixt_ti_l64_p16_ft384 --input_size 3 384 384 --pretrained --seed 42 --lr 2.5e-5 --sa_drop_path 0.05 --ca_drop_path 0.05 --workers 6 --torchcompile inductor --data_path $DATAPATH --batch_size_per_gpu 512 --epochs 30 --weight_decay 0.05 --sched cosine --reprob 0.0 --smoothing 0.0 --warmup_epochs 0 --drop 0.0 --opt lambc --mixup .8 --cutmix 1.0 --bce_loss --color_jitter 0.3 --three_augment --output_dir $OUTPUT_DIR
```
Make sure to pass the correct `model` name, `input_size` and `pretrained` flag as shown above. 
> [!NOTE]  
> In contrast to training from scratch, we used a **total batch-size** of 512 images for our finetuning experiments presented in the paper, as well as a smaller learning rate and no warmup steps (see hyperparameters above).
###Available weights of finetuned BiXT models
More insights regarding obtained ImageNet validation accuracies (top1 and top5), as well as the model weights are available [here](https://github.com/mrkshllr/BiXT/blob/main/MODEL_WEIGHTS.md#fine-tuned-bixt-tiny-models).


&nbsp;
## Evaluating BiXT models
To evaluate a [trained](https://github.com/mrkshllr/BiXT/blob/main/MODEL_WEIGHTS.md) BiXT model (here `bixt_ti_l64_p16`) on the ImageNet dataset, you can use the evaluation scripts provided in 
```
./experiment_scripts/eval_bixt_ti_l64_p16/
```
Make sure to define or replace:
* `$DATAPATH`: path to your ImageNet dataset (e.g. /mnt/datasets/ILSVRC2012)
* `$MODEL_CHECKPOINT`: checkpoint of the trained model to be evaluated (e.g. /mnt/bixt_checkpoints/model_best.pth)

The script then runs the evaluation procedure using:
```
python3 evaluate_BiXT.py --model bixt_ti_l64_p16 --model_checkpoint $MODEL_CHECKPOINT --workers 6 --data_path $DATAPATH --validation_batch_size 1024
```

You can also provide your wandb key, user and project name to upload the evaluation results (Acc@1, Acc@5 and Loss) to wandb.

&nbsp;

## Using BiXT for Dense Downstream Tasks
In the paper, we also presented results on a variety of dense downstream applications, such as semantic segmentation where a model pretrained on ImageNet is then finetuned on a task-specific dataset like ADE20K.

Note that for standard ImageNet training, we simply use a standard classification loss on the average-pooled latent embeddings for training. This means that for a 12 layer BiXT network, the refined patch tokens only receive a gradient until layer 11 -- which is why we employ only a *one-sided cross-attention* for the last layer (see BiXT model file [here](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L171)).

For simplicity and easy transfer to dense downstream tasks, we therefore simply create and train BiXT-models with a depth of 13 and train these on ImageNet (see [here](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L58)); Afterwards, the last one-sided cross-attention that exclusively refines the latent vectors is simply discarded and the remaining (fully-trained) 12-layer network is used for finetuning on downstream tasks. 

Note: It is, of course, entirely possible to replace or extend our simple classification loss on the averaged latent vectors through other token-side losses (e.g. Masked Image Modelling) to provide a gradient signal for the token side and thereby directly train both, the latent and token refinement for all layers.
###Available weights of pretrained dense 'backbone' BiXT models
More insights regarding obtained ImageNet validation accuracies (top1 and top5), as well as the model weights are available [here](https://github.com/mrkshllr/BiXT/blob/main/MODEL_WEIGHTS.md#bixt-model-weights-for-dense-downstream-tasks).


---
## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/mrkshllr/BiXT/blob/main/LICENSE) file.

---
## Citing BiXT
If you find this repository useful, please consider giving us a star :star: and cite our [work](https://arxiv.org/pdf/2402.12138):
```
@inproceedings{
    hiller2024bixt,
    title={Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers},
    author={Markus Hiller and Krista A. Ehinger and Tom Drummond},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    pages = {94097--94129},
    volume = {37},
    year={2024},
}
```
:point_right: If you have any questions regarding our work, please feel free to reach out! :email: 

---
For alternative implementations, please also check out [lucidrains' version](https://github.com/lucidrains/bidirectional-cross-attention) (also in PyTorch) and [axrwl's project](https://github.com/axrwl/bidirectional-cross-attention) for a JAX variant.