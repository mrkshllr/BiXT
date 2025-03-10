# BiXT Model Weights -- ImageNet
Model weights for the official PyTorch implementation of the paper **Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers** (NeurIPS 2024).
<div align="center">

:mortar_board: :page_facing_up: Find our paper: [[arXiv]](https://arxiv.org/pdf/2402.12138) &nbsp; [[NeurIPS]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ab1ee157f7804a13f980414b644a9460-Abstract-Conference.html) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; :milky_way: :chart_with_upwards_trend: [[Poster]](.github/Hiller_Poster_BiXT_NeurIPS2024_s.png) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; :mailbox_with_mail: Reference: [[BibTeX]](https://github.com/mrkshllr/BiXT#citing-bixt)
</div>

&nbsp;

## BiXT Model Weights -- ImageNet Classification
All models have been trained using a simple BCE loss for image classification on the ImageNet dataset, as detailed in the paper.

### Default BiXT-Tiny Models
Tiny BiXT models with 64 latents, using an embedding dimension of 192 and 12 layers:

| Model      |  arg: `model_name`  |   Acc@1   | Acc@5 |                               checkpoint                               |
|:-----------|:-----------------:|:---------:|:-----:|:----------------------------------------------------------------------:|
| BiXT-Ti/16 |  bixt_ti_l64_p16  | **80.10** |  **94.74**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882961) |
| BiXT-Ti/8  | bixt_ti_l64_p16s8 | **81.88** |  **95.43**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882994) |
| BiXT-Ti/4  | bixt_ti_l64_p16s4 | **82.70** |  **95.91**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882982) |

&nbsp;
### Fine-Tuned BiXT-Tiny Models
Tiny BiXT models finetuned on larger 384x384 images (for 30 epochs only):

| Model             |    arg: `model_name`     |   Acc@1   |   Acc@5   |                               checkpoint                               |
|:------------------|:------------------------:|:---------:|:---------:|:----------------------------------------------------------------------:|
| BiXT-Ti/16-ft384  |   bixt_ti_l64_p16_ft384  | **81.85** | **95.68** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882964) |
| BiXT-Ti/8-ft384   | bixt_ti_l64_p16s8_ft384  | **82.77** | **96.04** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882988) |
| BiXT-Ti/4-ft384   | bixt_ti_l64_p16s4_ft384  | **83.15** | **96.18** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882970) |

&nbsp;
### Convolutional Alternative: BiXT-Tiny w/ Conv-Tokeniser
Tiny BiXT model with convolutional instead of linear tokeniser, using 64 latents, an embedding dimension of 192 and 12 layers.  
Mainly used to gain insights regarding a different, more vision-specific tokeniser:

| Model             |        arg: `model_name`        |   Acc@1   | Acc@5 |  checkpoint  |
|:------------------|:-----------------------------:|:---------:|:-----:|:------------:|
| BiXT-Ti/16 (conv) | bixt_conv_ti_l64_p16 | **81.00** |  **95.18**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882946) |


&nbsp;
### Slightly larger model: Embedding dimension of 256 
BiXT model with increased embedding dimension of 256, using 64 latents and 12 layers.    
Mainly used to gain insights regarding scaling, including smaller patch size (8) and larger input images (384x384):

| Model              |        arg: `model_name`        |    Acc@1    |   Acc@5   |                               checkpoint                               |
|:-------------------|:-----------------------------:|:-----------:|:---------:|:----------------------------------------------------------------------:|
| BiXT-d256/16       |      bixt_ed256_l64_p16       |  **81.74**  | **95.37** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882976) |
| BiXT-d256/8        |    bixt_ed256LS_l64_p16s8*    |  **83.24**  | **96.17** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882991) |
| BiXT-d256/8-ft384  | bixt_ed256LS_l64_p16s8_ft384* | **83.89**   | **96.58** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882985) |

*these two models use layer-scale for improved stabilty in training.
&nbsp;  

---

## BiXT Model Weights for Dense Downstream Tasks

Note that for standard ImageNet training, we simply use a standard classification loss (BCE) on the average-pooled latent embeddings for training. This means that for a 12 layer BiXT network (see above), the refined patch tokens only receive a gradient until layer 11 -- which is why we employ only a *one-sided cross-attention* for the last layer (see BiXT model file [here](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L171)).

For simplicity and easy transfer to dense downstream tasks, we therefore simply create and train BiXT-models with a depth of 13 and train these on ImageNet (see [here](https://github.com/mrkshllr/BiXT/blob/main/timm/models/bixt.py#L58)); Afterwards, the last one-sided cross-attention that exclusively refines the latent vectors is simply discarded and the remaining (fully-trained) 12-layer network is used for finetuning on downstream tasks. 

Note: It is, of course, entirely possible to replace or extend our simple classification loss on the averaged latent vectors through other token-side losses (e.g. Masked Image Modelling) to provide a gradient signal for the token side and thereby directly train both, the latent and token refinement for all layers.

We provide a selection of **such pre-trained d13** models here:


### Dense (d13) BiXT-Tiny Models
Tiny dense BiXT models with 64 latents, using an embedding dimension of 192 and 13 layers during pretraining -- the last of which is to be dropped for dense downstream tasks, as discussed above.

| Model            |   arg: `model_name`   |   Acc@1   |   Acc@5   |                               checkpoint                               |
|:-----------------|:---------------------:|:---------:|:---------:|:----------------------------------------------------------------------:|
| BiXT-Ti/16 (d13) | bixt_ti_l64_d13_p16   | **80.44** | **94.90** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882958) |
| BiXT-Ti/8 (d13)  | bixt_ti_l64_d13_p16s8 |**82.44**  | **95.82** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882979) |
| BiXT-Ti/4 (d13)  | bixt_ti_l64_d13_p16s4 | **83.04** | **96.18** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882967) |

&nbsp;
### Fine-Tuned Dense (d13) BiXT-Tiny Models
Tiny dense BiXT models further finetuned on larger 384x384 images (for 30 epochs only):

| Model                    |       arg: `model_name`       |             Acc@1             | Acc@5 |                               checkpoint                               |
|:-------------------------|:---------------------------:|:-----------------------------:|:-----:|:----------------------------------------------------------------------:|
| BiXT-Ti/16-ft384 (d13)   | bixt_ti_l64_d13_p16_ft384   | **82.15** |  **95.72**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882955) |
| BiXT-Ti/8-ft384  (d13)   | bixt_ti_l64_d13_p16s8_ft384 | **83.06** |  **96.27**  | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882973) |

&nbsp;
### Dense Convolutional Alternative: BiXT-Tiny (d13) w/ Conv-Tokeniser
Tiny dense BiXT model with convolutional instead of linear tokeniser, using 64 latents, an embedding dimension of 192 and 13 layers during pretraining -- the last of which is to be dropped for dense downstream tasks, as discussed above.  
Mainly used to gain insights regarding a different, more vision-specific tokeniser:

| Model                  |     arg: `model_name`      |   Acc@1   |   Acc@5   |                               checkpoint                               |
|:-----------------------|:------------------------:|:---------:|:---------:|:----------------------------------------------------------------------:|
| BiXT-Ti/16 (conv, d13) | bixt_conv_ti_l64_d13_p16 | **80.95** | **95.05** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882949) |
| BiXT-Ti/8 (conv, d13)  | bixt_conv_ti_l64_d13_p8  | **82.47** | **95.68** | [zip-file](https://figshare.unimelb.edu.au/ndownloader/files/52882952) |



