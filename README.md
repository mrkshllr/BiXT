# BiXT - Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers 
Official PyTorch implementation of the paper **Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers**.

:mortar_board: :page_facing_up: Find our paper: [[arXiv]](https://arxiv.org/pdf/2402.12138) &nbsp;&nbsp;&nbsp;&nbsp;| &nbsp;&nbsp;&nbsp;&nbsp; 
:bookmark: Reference: [[BibTeX]](https://github.com/mrkshllr/BiXT#citing-bixt)

## Note :pencil: 
- :hourglass_flowing_sand: **Code and trained models coming soon...** :hourglass_flowing_sand: :computer:

## Updates :tada:
- September 26, 2024: **BiXT is accepted at NeurIPS 2024!** :fire: 

## TL;DR :eyes:
<div align="center">
  <img width="95%" alt="BiXT concept" src=".github/BiXT_visual.png">
</div>

**BiXT** is a novel bi-directional Transformer architecture which scales linearly with input size in terms of computational cost and memory consumption, but does not suffer the drop in performance or limitation to only one input modality seen with other efficient Transformer-based approaches. BiXT is inspired by the Perceiver architectures but replaces iterative attention with an efficient bi-directional cross-attention module in which input tokens and latent variables attend to each other simultaneously, leveraging a naturally emerging attention-symmetry between the two. This approach unlocks a key bottleneck experienced by Perceiver-like architectures and enables the processing and interpretation of both semantics ('what') and location ('where') to develop alongside each other over multiple layers -- allowing its direct application to dense and instance-based tasks alike. By combining efficiency with the generality and performance of a full Transformer architecture, BiXT can process longer sequences like point clouds, text or images at higher feature resolutions and achieves competitive performance across a range of tasks like point cloud part segmentation, semantic image segmentation, image classification, hierarchical sequence modeling and document retrieval. Our experiments demonstrate that BiXT models outperform larger competitors by leveraging longer sequences more efficiently on vision tasks like classification and segmentation, and perform on par with full Transformer variants on sequence modeling and document retrieval -- but require 28% fewer FLOPs and are up to 8.4x faster.


## Citing BiXT
If you find this repository useful, please consider giving us a star :star: and cite our [work](https://arxiv.org/pdf/2402.12138):
```
@inproceedings{
hiller2024bixt,
title={Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers},
author={Markus Hiller and Rongkai Ma and Mehrtash Harandi and Tom Drummond},
booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
year={2024},
url={https://openreview.net/forum?id=p_g2nHlMus}
}
```
If you have any questions regarding our work, please feel free to reach out!
