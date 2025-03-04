""" Positional embeddings for BiXT
Part of the implementation of paper 'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138

Partially based on/inspired by the implementation by Martin Krasser https://github.com/krasserm/perceiver-io ,
as well as the paper 'XCiT: Cross-Covariance Image Transformers'

Extended & modified by M. Hiller - Copyright 2024
"""
import torch
from torch import nn as nn


# =================================================================
# =================== Image-based 2D Encodings ====================
# =================================================================

# Note: We have a fixed input size throughout training, which allows us pre-compute the required variables and
#       simply use them at inference time -- this differs from tasks like detection/segmentation where input sizes vary!
class PositionalEncodingFourierXCiTEff(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    -> Modified, main part removed from forward since no need to recompute constant embeddings!
    """

    def __init__(self, token_grid_H, token_grid_W, hidden_dim=32, dim=768, temperature=10000, flatten=True):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * torch.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.flatten = flatten

        mask = torch.zeros(token_grid_H, token_grid_W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(),
                             pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(),
                             pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        # register encoding tensor to buffer to be moved with the module to GPU when moving to device!
        self.register_buffer("pos", pos)

    def forward(self):
        pos = self.token_projection(self.pos)
        if self.flatten:
            pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return pos

    def forward_add(self, x_patches):
        assert self.flatten, "For direct addition, flatten must be enabled."
        pos = self.token_projection(self.pos)
        pos = pos.flatten(1).transpose(0, 1)  # CHW -> NC
        return x_patches + pos.unsqueeze(0)


# Same functionality as our usually-used "efficient" version of additive projected encodings, but adaptively setting up
# the grid-sizes at inference time to accommodate changing input image sizes
class PositionalEncodingFourierAdpt(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    -> Slightly modified, also used in XCiT paper;
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000, flatten=True):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * torch.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.flatten = flatten

    def forward(self, hw_shape):
        mask = torch.zeros(*hw_shape).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(),
                             pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(),
                             pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        pos = self.token_projection(pos)
        if self.flatten:
            pos = pos.flatten(1).transpose(0, 1)  # CHW -> NC
        return pos.unsqueeze(0)
