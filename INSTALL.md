# Installation and ImageNet Dataset

## Prerequisites: Environment & Packages
Please install [PyTorch](https://pytorch.org/)  as appropriate for your system. This codebase has been developed with 
python 3.8.18, PyTorch 2.0.1, CUDA 11.8 and torchvision 0.15.2 with the use of an 
[anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html) environment.

While we have not (yet) actively tested newer variants of PyTorch and CUDA, we expect these to work in the same way and not cause any major issues. 
If you encounter any, please reach out!

> [!IMPORTANT]  
> While we make use of parts of the [timm](https://huggingface.co/docs/timm/index) library, we provide a **modified and extended** version [here](timm) as part of this repository. 
Therefore, please do **not** install timm within your environment, as this will likely interfere with the provided files.

To create an appropriate conda environment (after you have successfully downloaded and set up conda), run the following command:
```
conda env create -f reqsenv.yml
```
Or, if this doesn't work, try:
```
conda create --name bixt_env --file requirements.txt -c pytorch -c nvidia -c conda-forge
```
If this doesn't work either, simply create an empty environment and install the packages manually:
```
conda create --name bixt_env python=3.8.18
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install yaml pyyaml
```
If you use a previously created environment, please make sure that you do NOT already have timm installed.

&nbsp;

Activate your environment via
```
conda activate bixt_env
```

&nbsp;

Note: For ease of use, the activation of the environment can be set by adding the path to your anaconda installation to the [experiment scripts](experiment_scripts); just uncomment the respective lines at the beginning of the provided files and replace `<path_to_miniconda3>` accordingly:
```
echo "Starting up environment..."
source /<path_to_miniconda3>/bin/activate bixt_env
echo "... done!"
```
&nbsp;

----------

## The ImageNet Dataset
Note that the code in this repository has been developed for training on the ImageNet dataset.
Access to the dataset can be requested via creating an account on the [official ImageNet website](https://www.image-net.org/download.php), 
with more details on the most popular ILSVRC 2012 set [here](https://www.image-net.org/challenges/LSVRC/index.php).

Our code uses the dataloader from the [timm](timm/data/dataset_factory.py) library, and we access the ImageNet folders via the `torch/image_folder` variant (automatically selected if `--dataset` argument is set to `'imagenet'`, see [here](https://github.com/mrkshllr/BiXT/blob/main/train_BiXT.py#L515)); but other ways are possible as well. 

This dataloader supports a range of different folder structures for ImageNet; but also makes it easy to extend the code and train our models on various other datasets, if desired. 

&nbsp;

ImageNet reference: 
```
@article{ILSVRC,
    Author = {Olga Russakovsky* and Jia Deng* and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = {{ImageNet Large Scale Visual Recognition Challenge}},
    Year = {2015},
    journal   = {International Journal of Computer Vision (IJCV)},
    volume={115},
    number={3},
    pages={211-252}
}
```



