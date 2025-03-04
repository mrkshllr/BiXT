#!/usr/bin/env python3
"""
Official implementation of paper 'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138
Modifications/Extensions Copyright Markus Hiller 2024

Note: This training script is heavily based on the "ImageNet Training Script" from the timm library
    hacked together by / Copyright 2020 Ross Wightman;
    -- mixed with elements from the training script from the DeiT repository (Touvron et al.)
        (with original copyright (c) Meta Platforms, Inc. and affiliates.
    -> Restructured, modified & extended by Markus Hiller 2024
"""

import os
import argparse
import logging
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup
from timm.loss import SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from timm.layers import convert_sync_batchnorm

from utils import _args_to_yaml, _get_num_classes, _get_hash_from_args, _get_default_datasize

# === Check for fast, mixed-precision & distributed libraries (nvidia apex, etc.) ===
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

# === Try loading wandb for experiment tracking (default on) ===
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# === New PyTorch 2.0 features to speed up models ===
try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')


# Note: We're building the train script on the timm library to make use of their extensive capabilities regarding
# augmentations, optimisers and schedulers, but also to allow easy open-source integration later on!
def get_args_parser():
    parser = argparse.ArgumentParser(description='BiXT training and validation script using modified timm', add_help=False)

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    # Keep this argument outside the dataset group because it is positional.
    parser.add_argument('--data_path', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='imagenet',
                        help='dataset type + name (e.g. "<type>/<name>") (default: imagenet)')
    group.add_argument('--train_split', metavar='NAME', default='train',
                       help='dataset train split (default: train)')
    group.add_argument('--val_split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    group.add_argument('--dataset_download', action='store_true', default=False,
                       help='Allow download of dataset for torch/ and tfds/ datasets that support it.')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='bixt_ti_l64_p16', type=str, metavar='MODEL',
                       help='Name of model to train (default: "bixt_ti_l64_p16")')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--no_resume_opt', action='store_true', default=False,
                       help='Resume from checkpoint but with a freshly initialised optimiser -- e.g. finetuning.')
    group.add_argument('--num_classes', type=int, default=None, metavar='N',
                       help='number of label classes (Model default if None)')
    group.add_argument('--input_size', default=None, nargs=3, type=int,
                       metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224); '
                                             'uses model default if empty')
    group.add_argument('--crop_pct', default=0.875, type=float,
                       metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                       help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                       help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                       help='Image resize interpolation type (overrides model)')
    group.add_argument('--batch_size_per_gpu', type=int, default=1024, metavar='N',
                       help='Input batch size PER GPU for training (default: 1024)')
    group.add_argument('--validation_batch_size', type=int, default=None, metavar='N',
                       help='Validation batch size override (default: None)')
    group.add_argument('--grad_accum_steps', type=int, default=1, metavar='N',
                       help='The number of steps to accumulate gradients (default: 1)')
    group.add_argument('--model_kwargs', nargs='*', default={}, action=utils.ParseKwargs)

    # # scripting / codegen
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                                 help='torch.jit.script the full model')
    scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                                 help="Enable compilation w/ specified backend (default: inductor).")

    # Optimizer parameters
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='lambc', type=str, metavar='OPTIMIZER',
                       help='Optimizer (default: "lambc")')
    group.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                       help='Optimizer Epsilon (default: 1e-8 from deit)')
    group.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                       help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight_decay', type=float, default=0.05,
                       help='weight decay (default: 0.05)')
    group.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                       help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip_mode', type=str, default='norm',
                       help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer_decay', type=float, default=None,
                       help='layer-wise learning rate decay (default: None)')
    group.add_argument('--opt_kwargs', nargs='*', default={}, action=utils.ParseKwargs)

    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                       help='LR scheduler (default: "step"')
    group.add_argument('--sched_on_updates', action='store_true', default=False,
                       help='Apply LR scheduler step on update instead of epoch end.')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
                       help='learning rate, overrides lr-base if set (default: None). '
                            'NOTE: This disables learning rate scaling entirely -- simply uses provided lr.')
    group.add_argument('--lr_base', type=float, default=4e-3, metavar='LR',
                       help='base learning rate (lin. example): lr = lr_base * global_batch_size / base_size')
    group.add_argument('--lr_base_size', type=int, default=1024, metavar='DIV',
                       help='base learning rate batch size (divisor, was default: 256).')
    group.add_argument('--lr_base_scale', type=str, default='', metavar='SCALE',
                       help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
    group.add_argument('--lr_noise', type=float, nargs='+', default=None, metavar='pct, pct',
                       help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr_noise_pct', type=float, default=0.67, metavar='PERCENT',
                       help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr_noise_std', type=float, default=1.0, metavar='STDDEV',
                       help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr_cycle_mul', type=float, default=1.0, metavar='MULT',
                       help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr_cycle_decay', type=float, default=0.1, metavar='MULT',
                       help='amount to decay each learning rate cycle (default: 0.1); '
                            'Note: This corresponds to the decay_rate from the DeiT code-base!')
    group.add_argument('--lr_cycle_limit', type=int, default=1, metavar='N',
                       help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr_k_decay', type=float, default=1.0,
                       help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                       help='warmup learning rate (default: 1e-5)')
    group.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                       help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-5)')
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                       help='number of epochs to train (default: 300)')
    group.add_argument('--start_epoch', default=None, type=int, metavar='N',
                       help='manual epoch number (useful on restarts) - NOTE that this will overwrite the resume '
                            'epoch that can be aquired from the checkpoint!')
    group.add_argument('--decay_milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                       help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                       help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup_prefix', action='store_true', default=False,
                       help='Exclude warmup period from decay schedule.'),
    group.add_argument('--cooldown_epochs', type=int, default=0, metavar='N',
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')

    # Augmentation & regularization parameters
    group = parser.add_argument_group('Augmentation and regularization parameters')
    ######
    # Augmentation from DeiT-III -- not included in timm (natively), but added in this repo version!
    parser.add_argument('--three_augment', action='store_true', default=False,
                        help="Three augment method from DeiT-III paper -- Mutually exclusive with the auto-augment "
                             "method from timm! Note that this ignores all aa params like flip, etc.")  # 3augment
    parser.add_argument('--no_three_augment', action='store_false', dest='three_augment',
                        help="Deactivate three augment method from DeiT-III paper -- Mutually exclusive with the auto-"
                             "augment method from timm! Note that this ignores all aa params like flip, etc.")
    group.set_defaults(three_augment=True)
    # ## Two more augmentations used in DeiT, but not used in our experiments:
    parser.add_argument('--sr_crop', action='store_true', default=False,
                        help="Applies the 'simple random crop' introduced in the DeiT-III paper")  # simple random crop
    group.add_argument('--aug_repeats', type=float, default=0,
                       help='Number of augmentation repetitions (distributed training only) (default: 0)')
    ######
    group.add_argument('--no_aug', action='store_true', default=False,
                       help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                       help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                       help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability')
    group.add_argument('--color_jitter', type=float, default=0.3, metavar='PCT',
                       help='Color jitter factor (default: 0.3)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--bce_loss', action='store_true',
                       help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--no_bce_loss', action='store_false', dest='bce_loss',
                       help='Disable BCE loss w/ Mixup/CutMix use.')
    group.set_defaults(bce_loss=True)
    group.add_argument('--bce_target_thresh', type=float, default=None,
                       help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                       help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.8,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    group.add_argument('--cutmix', type=float, default=1.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    group.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup_prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup_switch_prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup_mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup_off_epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.0,
                       help='Label smoothing (default was 0.1)')
    group.add_argument('--train_interpolation', type=str, default='bicubic',
                       help='Training interpolation (random, bilinear, bicubic; deit default: "bicubic")')

    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                       help='Dropout rate (default: 0.)')
    parser.add_argument('--sa_drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--ca_drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    group.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                       help='Drop block rate (default: None)')

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=0, metavar='S',
                       help='random seed (default: 0)')
    group.add_argument('--worker_seeding', type=str, default='all',
                       help='worker seed mode (default: all)')
    group.add_argument('--log_interval', type=int, default=10, metavar='N',
                       help='how many batches to wait before logging training status')
    group.add_argument('--recovery_interval', type=int, default=0, metavar='N',
                       help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('--checkpoint-hist', type=int, default=3, metavar='N',
                       help='number of checkpoints to keep (default: 10)')
    group.add_argument('--workers', type=int, default=8, metavar='N',
                       help='how many training processes to use (default: 8)')
    group.add_argument('--save_images', action='store_true', default=False,
                       help='save images of input batches every log interval for debugging')
    group.add_argument('--amp', action='store_true',
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--no_amp', action='store_false', default=False, dest='amp',
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.set_defaults(amp=True)
    group.add_argument('--amp_dtype', default='float16', type=str,
                       help='lower precision AMP dtype (default: float16)')
    group.add_argument('--amp_impl', default='native', type=str,
                       help='AMP impl to use, "native" or "apex" (default: native)')
    group.add_argument('--amp_disable_tf32', action='store_true', default=False,
                       help='AMP impl: Disable use of tf32 dtype (note: disabling might slow down training on A100!)')
    group.add_argument('--no_ddp_bb', action='store_true', default=False,
                       help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--synchronize_step', action='store_true', default=False,
                       help='torch.cuda.synchronize() end of each step')
    group.add_argument('--pin_mem', action='store_true',
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no_pin_mem', action='store_true', dest='pin_mem',
                       help='Disable pinning CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.set_defaults(pin_mem=True)
    group.add_argument('--no_prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher -- Prefetcher uses async cuda streams to speed up loading.')
    group.add_argument('--output_dir', default='', type=str, metavar='PATH',
                       help='path to output folder (default: none, current dir)')
    group.add_argument('--eval_metric', default='acc1', type=str, metavar='EVAL_METRIC',
                       help='Best metric (default: "acc1"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int, help='Leave untouched. Internal use.')

    # Experiment tracking arguments
    group.add_argument('--log_wandb', action='store_true',
                       help='log training and validation metrics to wandb')
    group.add_argument('--wandb_projname', default='BiXT_runs', type=str,
                       help='project used for logging progress to wandb')

    return parser


def _parse_args():
    parser = argparse.ArgumentParser('BiXT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args


def main():
    utils.setup_default_logging()

    # ========== Obtain experiment settings from arg-parser ==========
    args = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)

    # ========== Set up cuda, initialise distributed training if multiple GPUs are used ==========
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not args.amp_disable_tf32
        torch.backends.cudnn.benchmark = True

    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability -> default 'on' as native
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    # Set seed for 'partial' reproducibility
    utils.random_seed(args.seed, args.rank)

    #
    # ========== Building model -- using the provided arguments ==========
    #
    if args.num_classes is None:  # Try to obtain classes for popular datasets if not specified via provided arguments.
        args.num_classes = _get_num_classes(args.dataset)
        assert args.num_classes is not None, "Number of classes for the specified dataset could not be retrieved. " \
                                             "Please specify via 'args.num_classes' directly!"
    if args.input_size is None:
        args.input_size = _get_default_datasize(args.dataset)
        assert args.input_size is not None, "Default input sizes for the specified dataset could not be retrieved. " \
                                            "Please specify via 'args.input_size' directly!"
    #

    if utils.is_primary(args):
        _logger.info(f"Creating model: {args.model}")
    if 'bixt' in args.model:  # Only support BiXT models for now -- can easily be extended to others if desired
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            sa_drop_path_rate=args.sa_drop_path,
            ca_drop_path_rate=args.ca_drop_path,
            drop_block_rate=None,
            checkpoint_path=args.initial_checkpoint
        )
    else:
        raise NotImplementedError("Model currently not supported.")

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} successfully created, '
            f'total param count incl classifier: {sum([m.numel() for m in model.parameters()])}')
        args.num_param_total = sum([m.numel() for m in model.parameters()])

    # move model to GPU
    model.to(device=device)

    # Only Prototypical: Set up synchronized BatchNorm for distributed training <- useful for ConvTok & LPI experiment,
    #  see 'xcit_lpi_convtok.py' file in timm/layers and 'bixt_ti_cnv_t64_p16.py' in timm/models
    if args.distributed and ('lpi' in args.model or 'cnv' in args.model):
        model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    #
    # ========== Preparing the optimiser(s) ==========
    #
    # Note: Setting the learning rate directly disables scaling of the learning rate!
    #       -> If scaling is desired, use the lr_base argument to provide base lr.
    if not args.lr:
        global_batch_size = args.batch_size_per_gpu * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling'
                f'and base reference batch size {args.lr_base_size} -- leading to a scaling ratio of {batch_ratio}.')

    # Save total batch size used for training to arguments list
    args.batch_size_total = args.batch_size_per_gpu * args.world_size * args.grad_accum_steps

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        if device.type == 'cuda' and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:  # Default in our experiments
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile)

    #
    # ========== Preparing the dataset and augmentation procedures ==========
    #
    # FIXME: Note -> This only really sets defaults for imagenet, not for other datasets! -> FIX if needed
    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # create the train and eval datasets
    if utils.is_primary(args):
        _logger.info("Creating data set...")
    ds_name = 'torch/image_folder' if args.dataset == 'imagenet' else args.dataset
    dataset_train = create_dataset(
        ds_name,
        root=args.data_path,
        split=args.train_split,
        is_training=True,
        download=args.dataset_download,
        batch_size=args.batch_size_per_gpu,
        seed=args.seed,
    )

    dataset_val = create_dataset(
        ds_name,
        root=args.data_path,
        split=args.val_split,
        is_training=False,
        download=args.dataset_download,
        batch_size=args.batch_size_per_gpu,
    )

    if utils.is_primary(args):
        _logger.info(f"Using {args.dataset} to run training for this experiment.\n"
                     f"Data successfully loaded: There are {len(dataset_train)} training images available.")

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    if utils.is_primary(args):
        _logger.info("Creating data loaders...")
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size_per_gpu,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        no_rr_crop=args.sr_crop,  # MH: Added simple random crop (deit-III paper), not used in our experiments though
        three_augment=args.three_augment,  # MH: Added "three_augment"  (deit-III paper)
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        persistent_workers=args.workers != 0,
    )

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # Reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_val,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or int(args.batch_size_per_gpu),  # * 1.5),
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
        device=device,
        persistent_workers=args.workers != 0,
    )

    # Note: While we only use BCE in our experiments,
    #       We support multiple other popular loss functions for easier experimentation / ablations
    # == Setup loss function (and store in args for easier readability later -- also use for output_dir) ==
    if args.bce_loss:  # >> Our default in BiXT for all models! <<
        train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        loss_nme = 'bce_loss'
    elif mixup_active and not args.bce_loss:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        train_loss_fn = SoftTargetCrossEntropy()
        loss_nme = 'stce_loss'
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        loss_nme = 'lsce_loss'
    else:
        train_loss_fn = nn.CrossEntropyLoss()
        loss_nme = 'ce_loss'
    args.loss_nme = loss_nme
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    #
    # ========== Setting up output dir, checkpoint saver and eval metric tracking ==========
    #
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        # If no output dir provided for training run, auto-create
        if args.output_dir == '':
            args.output_dir = './tmp_output/train'

        # Creating hash to uniquely identify parameter setting for run, but w/o elements that are non-essential and
        # might change due to moving the dataset, using different server, etc.
        non_essential_keys = ['data_path', 'dataset_download', 'resume', 'log_interval',
                              'recovery_interval', 'checkpoint_hist', 'workers', 'save_images', 'pin_mem',
                              'output_dir', 'log_wandb', 'wandb_projname']
        exp_hash = _get_hash_from_args(args, non_essential_keys)

        args.output_dir = os.path.join(args.output_dir, args.dataset + f"_{str(data_config['input_size'][-1])}",
                                       safe_model_name(args.model), f'{loss_nme}',
                                       f'bs{args.batch_size_total}_ep{args.epochs}', exp_hash)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        output_dir = args.output_dir

        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=None,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )

        # Store all updated arguments into args.yaml for later repeatability
        args_text = _args_to_yaml(args)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.wandb_projname, entity="your_username", config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    #
    # ========== Setting up learning rate scheduler(s) ==========
    #
    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )

    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch (!)
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch

    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args) and lr_scheduler is not None:
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')
    #
    # ========== Start the training procedure ==========
    #
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            # Run actual training for one epoch and obtain metrics
            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                mixup_fn=mixup_fn,
            )

            # Run validation on the validation set and obtain metrics
            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )

            # Log results to summary file and also upload to wandb (if enabled)
            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary_ours(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=(args.log_wandb and has_wandb),
                    num_params=args.num_param_total,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                if not torch.isnan(loss):  # Added to avoid the occasional NaN loss problem
                    _backward(loss)
        else:
            loss = _forward()
            if not torch.isnan(loss):   # Added to avoid the occasional NaN loss problem
                _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.2f} ({data_time_m.avg:.2f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta  # << Inactive in our experiments
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('acc1', top1_m.avg), ('acc5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    # And off we go...
    main()

    print("\n>> Congrats! Training your BiXT model has finished... happy evaluating!")
    print(">> If you like our work, please cite our paper 'https://arxiv.org/pdf/2402.12138'\n"
          ">> and consider giving us a star on github 'https://github.com/mrkshllr/BiXT'.")
