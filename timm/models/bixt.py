"""
BiXT Architecture Variants for Computer Vision applications;
Official implementation of paper 'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138
Extensions & Modifications Copyright Markus Hiller 2024

Note: this script is partially based on the 'Vision Transformer' implementation by Ross Wightman (timm library),
with original copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ._builder import build_model_with_cfg
from ._manipulate import named_apply
from timm.layers import DropPath, trunc_normal_, lecun_normal_
from timm.layers import PatchEmbed, PatchEmbedHalfStride, PatchEmbedQuarterStride
from ._registry import register_model
from timm.layers import PositionalEncodingFourierXCiTEff, PositionalEncodingFourierAdpt
from timm.layers import Mlp

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        **kwargs
    }


def _cfg_384(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 384, 384), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        **kwargs
    }


default_cfgs = {
    # default models w/ head-dimension of 32 (seems to work better for BiXT than 64) and 64 latent vectors:
    'bixt_ti_l64_p16': _cfg(),
    'bixt_ti_l64_p16s8': _cfg(),
    'bixt_ti_l64_p16s4': _cfg(),

    # 1 more layer (d13) for pre-training for dense token-processing applications, like sem-seg:
    'bixt_ti_l64_d13_p16': _cfg(),
    'bixt_ti_l64_d13_p16s8': _cfg(),
    'bixt_ti_l64_d13_p16s4': _cfg(),

    # Varying the number of latents
    'bixt_ti_l32_p16': _cfg(),
    'bixt_ti_l128_p16': _cfg(),
    'bixt_ti_l32_p16s8': _cfg(),
    'bixt_ti_l128_p16s8': _cfg(),

    # Larger models, i.e. larger embedding dimension
    'bixt_ed256_l64_p16': _cfg(),
    'bixt_ed256_l64_p16s8': _cfg(),
    'bixt_ed256LS_l64_p16s8': _cfg(),  # layer-scale can sometimes help for larger archs w/ long sequences

    # TODO: INSERT YOUR PATHS TO PRETRAINED MODELS HERE as 'file' (e.g. weights downloaded from our git repository)
    # Finetuning default models (224x224) on higher resolution of 384x384
    'bixt_ti_l64_p16_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),
    'bixt_ti_l64_p16s8_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),
    'bixt_ti_l64_p16s4_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),

    # Finetuning these d13 models on higher resolution of 384x384
    'bixt_ti_l64_d13_p16_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),
    'bixt_ti_l64_d13_p16s8_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),

    # larger models fine-tuned on higher resolution of 384x384:
    'bixt_ed256_l64_p16s8_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),
    'bixt_ed256LS_l64_p16s8_ft384': _cfg_384(file='<Path_to_pretrained_model>/model_best.pth.tar'),
 }


class SelfAttention(nn.Module):
    """Self-Attention for refining the latents """
    def __init__(self, dim, dim_attn, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.qkv = nn.Linear(dim, dim_attn * 3, bias=qkv_bias)  # projecting to 'inner' attention emb_dim
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim)    # re-projecting to 'outer' original emb_dim of latents
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_attn)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Cross-Attention between latents and input tokens -- returning the refined latents and tokens as tuple """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.rv_latents = nn.Linear(dim_lat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)  # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_dropT = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)             # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)
        self.proj_pat = nn.Linear(dim_attn, dim_pat)             # 'out-projection' for patches/tokens
        self.proj_drop_pat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape  # Note: need B_lat since 1 at very first pass, then broadcasted/extended to bs
        B_pat, N_pat, _ = x_patches.shape
        rv_lat = self.rv_latents(x_latents).reshape(B_lat, N_lat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_lat, v_lat = rv_lat.unbind(0)
        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale  # query from latent, key from patches
        attn_T = attn.transpose(-2, -1)  # bidirectional attention, associated with the values from the query q

        attn = attn.softmax(dim=-1)  # softmax along patch token dimension
        attn_T = attn_T.softmax(dim=-1)  # softmax along latent token dimension

        attn = self.attn_drop(attn)
        attn_T = self.attn_dropT(attn_T)

        # Retrieve information form the patch tokens via latent query:
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        # Likewise, store information from the latents in the patch tokens via transposed attention:
        x_patches = (attn_T @ v_lat).transpose(1, 2).reshape(B_pat, N_pat, self.dim_attn)
        x_patches = self.proj_pat(x_patches)
        x_patches = self.proj_drop_pat(x_patches)

        return x_latents, x_patches


class CrossAttentionOneSided(nn.Module):
    """Cross-Attention between latents and input tokens -- only returning the refined latents here, used at the last
        stage of the BiXT (since we don't use the patch tokens afterwards, we can save the compute) """
    def __init__(self, dim_lat, dim_pat, dim_attn, num_heads=8, rv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim_attn % num_heads == 0, 'dim_attn MUST be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim_attn // num_heads
        self.scale = head_dim ** -0.5
        self.dim_attn = dim_attn

        self.r_latents = nn.Linear(dim_lat, dim_attn, bias=rv_bias)             # 'in-projection' for latents
        self.rv_patches = nn.Linear(dim_pat, dim_attn * 2, bias=rv_bias)        # 'in-projection' for patches/tokens
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_lat = nn.Linear(dim_attn, dim_lat)                            # 'out-projection' for latents
        self.proj_drop_lat = nn.Dropout(proj_drop)

    def forward(self, x_latents, x_patches):
        B_lat, N_lat, _ = x_latents.shape  # Note: need B_lat since 1 at very first pass, then broadcasted/extended to bs
        B_pat, N_pat, _ = x_patches.shape
        r_lat = self.r_latents(x_latents).reshape(B_lat, N_lat, 1, self.num_heads,
                                                  self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)

        rv_pat = self.rv_patches(x_patches).reshape(B_pat, N_pat, 2, self.num_heads,
                                                    self.dim_attn // self.num_heads).permute(2, 0, 3, 1, 4)
        r_pat, v_pat = rv_pat.unbind(0)
        # attention: (q@k.T), and will be multiplied with the value associated with the keys k
        attn = (r_lat @ r_pat.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)   # softmax along patch token dimension
        attn = self.attn_drop(attn)

        # Retrieve information form the patch tokens via latent query:
        x_latents = (attn @ v_pat).transpose(1, 2).reshape(-1, N_lat, self.dim_attn)
        x_latents = self.proj_lat(x_latents)
        x_latents = self.proj_drop_lat(x_latents)

        return x_latents


# LayerScale NOT used by default, but might be beneficial for larger / deeper models
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# Self-Attention Block for latent tokens
class SABlock(nn.Module):
    """Block performing Self-Attention over the latents"""
    def __init__(
            self, dim_lat, dim_attn, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_lat)
        self.attn = SelfAttention(dim=dim_lat, dim_attn=dim_attn, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, papers/repos indicate that that's better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim_lat)
        self.mlp = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * mlp_ratio), act_layer=act_layer,
                           drop=drop)

        self.ls2 = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Cross-Attention Block for latent tokens and patch tokens, bi-directional attention
class CABlock(nn.Module):
    """Block performing Cross-Attention between the latents and input tokens, bi-directional attention"""
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4., pat_mlp_ratio=4.):
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = CrossAttention(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                   rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.ls1_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, papers/repos indicate that that's better than dropout here
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Latents -- components for further refinement after attention:
        self.norm2_lat = norm_layer(dim_lat)
        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)

        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Patches -- components for further refinement after attention:
        self.norm2_pat = norm_layer(dim_pat)
        self.mlp_pat = Mlp(in_features=dim_pat, hidden_features=int(dim_pat * pat_mlp_ratio),
                           act_layer=act_layer, drop=drop)

        self.ls2_pat = LayerScale(dim_pat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_pat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_latents, x_patches):
        # Cross attention forwards
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat, x_pat = self.attn(x_lat, x_pat)

        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))

        x_patches = x_patches + self.drop_path1_pat(self.ls1_pat(x_pat))
        x_patches = x_patches + self.drop_path2_pat(self.ls2_pat(self.mlp_pat(self.norm2_pat(x_patches))))

        return x_latents, x_patches


class CAOneSidedBlock(nn.Module):
    """Block performing one-sided Cross-Attention between the latents and input tokens, no information transfer
       to input tokens for this block!"""
    def __init__(
            self, dim_lat, dim_pat, dim_attn, num_heads, rv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            lat_mlp_ratio=4.):
        super().__init__()
        self.norm1_lat = norm_layer(dim_lat)
        self.norm1_pat = norm_layer(dim_pat)
        self.attn = CrossAttentionOneSided(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn, num_heads=num_heads,
                                           rv_bias=rv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # Latents -- components for further refinement after attention:
        self.norm2_lat = norm_layer(dim_lat)

        self.mlp_lat = Mlp(in_features=dim_lat, hidden_features=int(dim_lat * lat_mlp_ratio),
                           act_layer=act_layer, drop=drop)
        self.ls2_lat = LayerScale(dim_lat, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_lat = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_latents, x_patches):
        # Cross attention forwards
        x_lat = self.norm1_lat(x_latents)
        x_pat = self.norm1_pat(x_patches)
        x_lat = self.attn(x_lat, x_pat)

        x_latents = x_latents + self.drop_path1_lat(self.ls1_lat(x_lat))
        x_latents = x_latents + self.drop_path2_lat(self.ls2_lat(self.mlp_lat(self.norm2_lat(x_latents))))

        return x_latents, None


class BiXTBlock(nn.Module):
    # Note: dim_lat = latent dimension,
    #       dim_pat = patch token or other input dimension,
    #       dim_attn = inner dimension of tensors which is used to cast the attention between query and keys

    def __init__(
            self, dim_lat, dim_pat, dim_attn, sa_num_heads, ca_num_heads, sa_mlp_ratio=4.,
            qkv_bias=False, rv_bias=False, sa_drop=0., ca_drop=0., sa_attn_drop=0., ca_attn_drop=0., init_values=None,
            sa_drop_path=0.10, ca_drop_path=0.10, act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_onesided=False,
            ca_lat_mlp_ratio=4., ca_pat_mlp_ratio=4.):
        super().__init__()
        self.self_attention_latents = SABlock(dim_lat=dim_lat, dim_attn=dim_lat, num_heads=sa_num_heads,
                                              mlp_ratio=sa_mlp_ratio, qkv_bias=qkv_bias,
                                              drop=sa_drop, attn_drop=sa_attn_drop, init_values=init_values,
                                              drop_path=sa_drop_path, act_layer=act_layer, norm_layer=norm_layer)
        if not ca_onesided:
            self.cross_attention_latents_token = CABlock(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn,
                                                         num_heads=ca_num_heads,
                                                         rv_bias=rv_bias, drop=ca_drop, attn_drop=ca_attn_drop,
                                                         init_values=init_values, drop_path=ca_drop_path,
                                                         act_layer=act_layer, norm_layer=norm_layer,
                                                         lat_mlp_ratio=ca_lat_mlp_ratio,
                                                         pat_mlp_ratio=ca_pat_mlp_ratio)
        else:
            self.cross_attention_latents_token = CAOneSidedBlock(dim_lat=dim_lat, dim_pat=dim_pat, dim_attn=dim_attn,
                                                                 num_heads=ca_num_heads, lat_mlp_ratio=ca_lat_mlp_ratio,
                                                                 rv_bias=rv_bias, drop=ca_drop, attn_drop=ca_attn_drop,
                                                                 init_values=init_values, drop_path=ca_drop_path,
                                                                 act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x_lat_pat):
        # Input needs to be passed as tuple since nn.Sequential doesn't allow multiple inputs
        # Tuple x_lat_pat: (x_latents, x_patches)
        # Perform forward pass through cross attention module, relating latents and patch tokens
        x_latents, x_patches = self.cross_attention_latents_token(*x_lat_pat)

        # Perform self-attention to refine the latent embeddings
        x_latents = self.self_attention_latents(x_latents)

        return x_latents, x_patches


class BiXT(nn.Module):
    """ Bidirectional Cross-Attention Transformer architecture
    """
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_latents=64, global_pool='avg',
            embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=None, depth=12, sa_num_heads=6, ca_num_heads=6,
            sa_mlp_ratio=4., qkv_bias=True, rv_bias=True, init_values=None, fc_norm=True,
            sa_drop_rate=0., ca_drop_rate=0., sa_attn_drop_rate=0., ca_attn_drop_rate=0.,
            sa_drop_path_rate=0., ca_drop_path_rate=0., weight_init='jax', embed_layer=PatchEmbed, norm_layer=None,
            act_layer=None, block_fn=BiXTBlock, latinit=True,
            posenc_type_pat='fourier2d_embadd_proj', ca_lat_mlp_ratio=4., ca_pat_mlp_ratio=4.,
            **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            num_latents (int): number of latent vectors (trainable)
            global_pool (str): type of global pooling for final sequence (default: 'avg')
            embed_dim_lat (int): embedding dimension for latent vectors
            embed_dim_pat (int): embedding dimension patch-tokens
            embed_dim_attn (int): embedding dimension for the attention operation (could differ from lat/pat,
                                  but is identical by default)
            depth (int): depth of transformer
            sa_num_heads (int): number of self-attention heads
            ca_num_heads (int): number of cross-attention heads
            sa_mlp_ratio (int): ratio of mlp hidden dim to embedding dim for self-attention block
            qkv_bias (bool): enable bias for qkv if True
            rv_bias (bool): enable bias for rv if True
            init_values: (float): layer-scale init values; default=None (deactivated)
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            sa_drop_rate (float): dropout rate self-attention
            ca_drop_rate (float): dropout rate cross-attention
            sa_attn_drop_rate (float): attention dropout rate self-attention
            ca_attn_drop_rate (float): attention dropout rate cross-attention
            sa_drop_path_rate (float): stochastic depth rate self-attention
            ca_drop_path_rate (float): stochastic depth rate cross-attention
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            block_fn: (nn.Module): function definition for block, default=BiXTBlock
            latinit: (bool): initialising the latent vectors (default=True)
            posenc_type_pat: (str): type of positional encoding for patch tokens (default='fourier2d_embadd_proj')
            ca_lat_mlp_ratio: (int): ratio of mlp hidden dim to embedding dim for cross-attention block latents
            ca_pat_mlp_ratio: (int): ratio of mlp hidden dim to embedding dim for cross-attention block patch-tokens
        """
        super().__init__()
        assert global_pool == 'avg', "Currently only 'avg' supported for global pooling."
        use_fc_norm = fc_norm  # Usually goes along with average pooling in ViTs, i.e. default True
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # Default attention-dim to the dimension of the latents
        embed_dim_attn = embed_dim_lat if embed_dim_attn is None else embed_dim_attn
        self.embed_dim_pat = embed_dim_pat
        self.num_features = self.embed_dim_lat = embed_dim_lat  # num_features of latents for consistency (other models)

        self.num_classes = num_classes
        self.global_pool = global_pool  # currently, only average 'avg' is supported

        # Potential dropout (default disabled)
        self.pos_drop_pat = nn.Dropout(p=ca_drop_rate)
        self.pos_drop_lat = nn.Dropout(p=sa_drop_rate)
        dpr_pat = [x.item() for x in torch.linspace(0, ca_drop_path_rate, depth)]  # stochastic depth decay rule
        dpr_lat = [x.item() for x in torch.linspace(0, sa_drop_path_rate, depth)]  # stochastic depth decay rule

        # ================= INPUT/PATCHES ================= #
        # Patch tokens: Add positional encodings and project to embedding space
        # >> Approaches to positional encoding:
        if embed_layer in (PatchEmbed, PatchEmbedHalfStride, PatchEmbedQuarterStride):
            if posenc_type_pat == 'fourier2d_embadd_proj':
                self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                               in_chans=in_chans,
                                               embed_dim=embed_dim_pat, flatten=True, **kwargs)
                # Using the posenc from XCiT, DETR, etc. -> fixed hidden dim that's up-projected to emb_dim;
                # Emulates the original posemb from 'Attention is all you need' paper, w/ default temp 10k
                self.fourier_pos_encode = \
                    PositionalEncodingFourierXCiTEff(token_grid_H=self.patch_embed.grid_size[0],
                                                     token_grid_W=self.patch_embed.grid_size[1],
                                                     hidden_dim=32, dim=embed_dim_pat,
                                                     temperature=10000, flatten=True)
                self.encode_pat = self._encode_pat_2dfourier_emb_add
                # Note: the forward of this pos_enc only returns the actual embedding, does not (yet) directly add it
            elif posenc_type_pat == 'fourier2d_embadd_proj_dyn':  # Same as before, but dynamic size at forward pass!
                self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                               in_chans=in_chans,
                                               embed_dim=embed_dim_pat, flatten=True, **kwargs)
                # Using the posenc from XCiT, DETR, etc. -> fixed hidden dim that's upprojected to emb_dim;
                # Emulates the original posemb from 'Attention is all you need' paper, w/ default temp 10k
                self.fourier_pos_encode = \
                    PositionalEncodingFourierAdpt(hidden_dim=32, dim=embed_dim_pat,
                                                  temperature=10000, flatten=True)
                self.encode_pat = self._encode_pat_2dfourier_emb_add_adpt
                # Note: the forward of this pos_enc only returns the actual embedding, does not (yet) directly add it
            else:
                raise NotImplementedError("Requested positional embedding for input data currently not supported.")
        else:
            raise NotImplementedError("Requested tokenisation method for input data currently not supported.")
        self.posenc_type_pat = posenc_type_pat

        # ==================== LATENTS ==================== #
        # Latents: Create
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim_lat))
        self.latinit = latinit  # initialise latents in common way (trunc_normal_(0,0.02) , or not

        # Note: We have bidirectional cross attention for the d-1 first layers, and one-sided cross-attention for the
        # last layer since we are only using the latents to classify, and do not require an update to the patch tokens
        # in our default setup -- this might be changed in case patch-based losses are engaged, like MIM or similar
        self.blocks = nn.Sequential(*[
            block_fn(
                dim_lat=embed_dim_lat, dim_pat=embed_dim_pat, dim_attn=embed_dim_attn,
                sa_num_heads=sa_num_heads, ca_num_heads=ca_num_heads,
                sa_mlp_ratio=sa_mlp_ratio, qkv_bias=qkv_bias, rv_bias=rv_bias,
                init_values=init_values, sa_drop=sa_drop_rate, ca_drop=ca_drop_rate,
                sa_attn_drop=sa_attn_drop_rate, ca_attn_drop=ca_attn_drop_rate,
                sa_drop_path=dpr_lat[i], ca_drop_path=dpr_pat[i], norm_layer=norm_layer, act_layer=act_layer,
                ca_lat_mlp_ratio=ca_lat_mlp_ratio, ca_pat_mlp_ratio=ca_pat_mlp_ratio,
            )
            if i != depth-1 else
            block_fn(
                dim_lat=embed_dim_lat, dim_pat=embed_dim_pat, dim_attn=embed_dim_attn,
                sa_num_heads=sa_num_heads, ca_num_heads=ca_num_heads,
                sa_mlp_ratio=sa_mlp_ratio, qkv_bias=qkv_bias, rv_bias=rv_bias,
                init_values=init_values, sa_drop=sa_drop_rate, ca_drop=ca_drop_rate,
                sa_attn_drop=sa_attn_drop_rate, ca_attn_drop=ca_attn_drop_rate,
                sa_drop_path=dpr_lat[i], ca_drop_path=dpr_pat[i], norm_layer=norm_layer, act_layer=act_layer,
                ca_onesided=True, ca_lat_mlp_ratio=ca_lat_mlp_ratio, ca_pat_mlp_ratio=ca_pat_mlp_ratio,
            )
            for i in range(depth)
            ])
        self.norm_lat = norm_layer(embed_dim_lat) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim_lat) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim_lat, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        # Initialising the learnt latent vectors (True by default)
        if self.latinit:
            trunc_normal_(self.latents, std=.02)
        named_apply(get_init_weights_bixt(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_bixt_timm(m)

    def _encode_pat_2dfourier_emb_add(self, x_patches):
        x_patches = self.patch_embed(x_patches)             # Project input data into patched embeddings, flatten
        x_patches = self.fourier_pos_encode.forward_add(x_patches)  # Add fourier encodings to raw input data, flattened
        # Note: Patch_embed already flattens into sequence, so no further action required here
        return self.pos_drop_pat(x_patches)

    def _encode_pat_2dfourier_emb_add_adpt(self, x_patches):
        x_patches = self.patch_embed(x_patches)  # Project input data into patched embeddings, flatten
        pos_enc = self.fourier_pos_encode(hw_shape=self.patch_embed.grid_size)
        # Note: Patch_embed already flattens into sequence, so no further action required here
        #       Equally, pos_embed flattens and reshapes appropriately, so simply adding suffices
        x_patches = x_patches + pos_enc  # Broadcasts pos_enc across batch dim
        return self.pos_drop_pat(x_patches)

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x_patches):
        x_patches = self.encode_pat(x_patches)  # Embed and positionally encode input data, depending on arguments
        x_latents = self.latents  # Encode latents: either add pos encodings, or do nothing (default)
        x_latents, _ = self.blocks((x_latents, x_patches))
        x_latents = self.norm_lat(x_latents)
        return x_latents

    def forward_head(self, x_latents, pre_logits: bool = False):
        assert self.global_pool == 'avg'
        x_latents = x_latents.mean(dim=1)
        x_latents = self.fc_norm(x_latents)
        return x_latents if pre_logits else self.head(x_latents)

    def forward(self, x, pre_logits: bool = False):
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=pre_logits)
        return x


def init_weights_bixt_timm(module: nn.Module, name: str = ''):
    """ Alternative weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_bixt_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ BiXT weight initialization, matching JAX (Flax) impl -- default for BiXT"""
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_bixt_moco(module: nn.Module, name: str = ''):
    """ Alternative weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_bixt(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:  # <== Default for BiXT
        return partial(init_weights_bixt_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_bixt_moco
    else:
        return init_weights_bixt_timm


def _create_vision_bixt(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        BiXT, variant, pretrained,
        **kwargs)
    return model


def pop_posenc_pos_ft384(state_dict, model):
    # Try to pop the pos buffer, since it no longer fits
    try:
        state_dict.pop('fourier_pos_encode.pos')
        _logger.info("Removed pos buffer from fourier positional encoding for compatibility with higher resolution.")
    except KeyError:
        _logger.warning("Could not pop positional encoding pos buffer -- make sure to actually load correct weights.")
    return state_dict


# #################################################################################################
# ########################  MODEL DEFINITIONS FOR EASY CREATION VIA STRING ########################
###################################################################################################

###################################################################################################
# ######## >>>>>>>> Default models with different patch/tokenisation sizes <<<<<<< ################
###################################################################################################
@register_model
def bixt_ti_l64_p16(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16', pretrained=pretrained, **model_kwargs)
    return model


# >> p16s8
@register_model
def bixt_ti_l64_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16s8', pretrained=pretrained, **model_kwargs)
    return model


# >> p16s4
@register_model
def bixt_ti_l64_p16s4(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/4)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedQuarterStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16s4', pretrained=pretrained, **model_kwargs)
    return model


# #### >>>>>>>> Fine Tuning models with different patch/tokenisation sizes <<<<<<< ################
@register_model
def bixt_ti_l64_p16_ft384(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/16-ft384)
    """
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj_dyn', weight_init='skip',
                        pretrained_strict=True,
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16_ft384', pretrained=pretrained,
                                pretrained_filter_fn=pop_posenc_pos_ft384, **model_kwargs)
    return model


# >> p16s4
@register_model
def bixt_ti_l64_p16s8_ft384(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/8-ft384)
    """
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj_dyn', weight_init='skip',
                        pretrained_strict=True, embed_layer=PatchEmbedHalfStride,
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16s8_ft384_kj', pretrained=pretrained,
                                pretrained_filter_fn=pop_posenc_pos_ft384, **model_kwargs)
    return model


# >> p16s4
@register_model
def bixt_ti_l64_p16s4_ft384(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/4-ft384)
    """
    model_kwargs = dict(img_size=384, patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj_dyn', weight_init='skip',
                        pretrained_strict=True, embed_layer=PatchEmbedQuarterStride,
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_p16s4_ft384', pretrained=pretrained,
                                pretrained_filter_fn=pop_posenc_pos_ft384, **model_kwargs)
    return model


###################################################################################################
# ######## >>>>>>>> 13-layer models with different patch/tokenisation sizes <<<<<<< ###############
###################################################################################################
@register_model
def bixt_ti_l64_d13_p16(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-d13/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=13,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_d13_p16', pretrained=pretrained, **model_kwargs)
    return model


# >> p16s8
@register_model
def bixt_ti_l64_d13_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-d13/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=13,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_d13_p16s8', pretrained=pretrained, **model_kwargs)
    return model


# >> p16s4
@register_model
def bixt_ti_l64_d13_p16s4(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-d13/4)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=13,
                        num_latents=64, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedQuarterStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l64_d13_p16s4', pretrained=pretrained, **model_kwargs)
    return model


###################################################################################################
# ############ >>>>>>>> Example models with more/fewer latent vectors <<<<<<< #####################
###################################################################################################

# 32 Latents p16
@register_model
def bixt_ti_l32_p16(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti32/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=32, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l32_p16', pretrained=pretrained, **model_kwargs)
    return model


# 32 Latents p16s8
@register_model
def bixt_ti_l32_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti32/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=32, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l32_p16s8', pretrained=pretrained, **model_kwargs)
    return model


# 128 Latents p16
@register_model
def bixt_ti_l128_p16(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti128/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=128, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        **kwargs)
    model = _create_vision_bixt('bixt_ti_l128_p16', pretrained=pretrained, **model_kwargs)
    return model


# 128 Latents p16s8
@register_model
def bixt_ti_l128_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti128/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=192, embed_dim_pat=192, embed_dim_attn=192, depth=12,
                        num_latents=128, sa_num_heads=6, ca_num_heads=6, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride, **kwargs)
    model = _create_vision_bixt('bixt_ti_l128_p16s8', pretrained=pretrained, **model_kwargs)
    return model


###################################################################################################
# ####### >>>>>>>> Larger models, i.e. larger dimension of 256 instead of 192 <<<<<<< #############
###################################################################################################

# 64 Latents, model emb-dim 256
@register_model
def bixt_ed256_l64_p16(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-e256/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=256, embed_dim_pat=256, embed_dim_attn=256, depth=12,
                        num_latents=64, sa_num_heads=8, ca_num_heads=8, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        **kwargs)
    model = _create_vision_bixt('bixt_ed256_l64_p16', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def bixt_ed256_l64_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-e256/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=256, embed_dim_pat=256, embed_dim_attn=256, depth=12,
                        num_latents=64, sa_num_heads=8, ca_num_heads=8, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride,
                        **kwargs)
    model = _create_vision_bixt('bixt_ed256_l64_p16s8', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def bixt_ed256LS_l64_p16s8(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti-e256/8)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=256, embed_dim_pat=256, embed_dim_attn=256, depth=12,
                        num_latents=64, sa_num_heads=8, ca_num_heads=8, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj', weight_init='jax',
                        embed_layer=PatchEmbedHalfStride, init_values=1e-5,
                        **kwargs)
    model = _create_vision_bixt('bixt_ed256LS_l64_p16s8', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def bixt_ed256_l64_p16s8_ft384(pretrained=False, **kwargs):
    """ Vision BiXT-Ti (BiXT-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim_lat=256, embed_dim_pat=256, embed_dim_attn=256, depth=12,
                        num_latents=64, sa_num_heads=8, ca_num_heads=8, qkv_bias=True, rv_bias=True, latinit=True,
                        posenc_type_pat='fourier2d_embadd_proj_dyn', weight_init='skip',
                        pretrained_strict=True,
                        embed_layer=PatchEmbedHalfStride,
                        **kwargs)
    model = _create_vision_bixt('bixt_ed256_l64_p16s8_ft384', pretrained=pretrained,
                                pretrained_filter_fn=pop_posenc_pos_ft384, **model_kwargs)
    return model
