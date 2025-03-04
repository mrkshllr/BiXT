""" Components originate from the XCiT repository --
Original copyright notice there:

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

Extended & modified by M. Hiller - 2024 as part of the paper
  'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138
Modifications & extensions Copyright Markus Hiller, 2024
"""

import logging
import torch
from torch import nn as nn
from .helpers import to_2tuple

_logger = logging.getLogger(__name__)


# ========================== CONV and LPI Module ======================================
# We only try this to show potential boost
# when incorporating more complex 'input-data specific' components into the architecture

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        # nn.SyncBatchNorm(out_planes)
        nn.BatchNorm2d(out_planes)
    )


class ConvPatchEmbed(nn.Module):
    """ Image to Patch Embedding using multiple convolutional layers
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise ValueError("For convolutional projection, patch size has to be in [4, 8, 16]")

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        assert Hp == self.grid_size[0] and Wp == self.grid_size[1], "Mismatch in feature map size! Please check"
        x = x.flatten(2).transpose(1, 2)
        return x


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communication performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, kernel_size=3, norm='bn'):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        if norm == 'bn':
            # self.bn = nn.SyncBatchNorm(in_features)
            self.bn = nn.BatchNorm2d(in_features)
            self.forward_opt = self.forward_bn
        elif norm == 'ln':
            self.bn = nn.LayerNorm(in_features)
            self.forward_opt = self.forward_ln
        else:
            raise KeyError("Selected type for LPI norm not implemented.")
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, H, W):
        x = self.forward_opt(x, H, W)
        return x

    def forward_ln(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.bn(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

    def forward_bn(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x
