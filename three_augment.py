"""
>> This script implements three different kinds of Image Augmentations that have been proposed for ImageNet Training
   as part of the '3Augment' strategy in 'DeiT III: Revenge of the ViT' by Touvron et al. (ECCV 2022), with original
   Copyright (c) Meta Platforms, Inc. and affiliates.

We have integrated these into the augmentation pipeline of the modified 'timm' library within this repository;
Please check the timm/data/transforms_factory.py file to see where these augmentations are used!

>> This file is part of the official implementation of the paper
   'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers' (Hiller et al.)
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138

Extensions/Modifications Copyright 2024 Markus Hiller
"""

from torchvision import transforms
import random
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Applying Gaussian Blur as augmentation with probability p, and given a uniformly sampled radius range.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img


class Solarization(object):
    """
    Applying solarization as augmentation with probability p.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class gray_scale(object):
    """
    Applying gray-scale augmentation with probability p.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


