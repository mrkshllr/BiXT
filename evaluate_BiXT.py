#!/usr/bin/env python3
"""
Official implementation of paper 'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138
Modifications/Extensions Copyright Markus Hiller 2024

Note: This validation script is based on parts of the "ImageNet Training Script" from the timm library
    hacked together by / Copyright 2020 Ross Wightman;
    -> Restructured, modified & extended by Markus Hiller 2024
"""

import os
import argparse
import logging
import time
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name
from timm.layers import convert_sync_batchnorm

from utils import _get_num_classes,  _get_default_datasize

# === Try loading wandb for experiment tracking (default on) ===
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

_logger = logging.getLogger('validation')


# Note: The validation script is built on our training script for ease of use and understanding.
def get_args_parser():
    parser = argparse.ArgumentParser(description='BiXT validation script on ImageNet', add_help=False)

    # Dataset parameters
    group = parser.add_argument_group('Dataset parameters')
    # Keep this argument outside the dataset group because it is positional.
    parser.add_argument('--data_path', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='imagenet',
                        help='dataset type + name (e.g. "<type>/<name>") (default: imagenet)')
    group.add_argument('--val_split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    group.add_argument('--dataset_download', action='store_true', default=False,
                       help='Allow download of dataset for torch/ and tfds/ datasets that support it.')

    # Model parameters
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='bixt_ti_t64_p16', type=str, metavar='MODEL',
                       help='Name of model to train (default: "bixt_ti_t64_p16")')
    group.add_argument('--model_checkpoint', default='', type=str, metavar='PATH',
                       help='Load model weights from this checkpoint -- required for evaluation!')
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
    group.add_argument('--validation_batch_size', type=int, default=1024, metavar='N',
                       help='Validation batch size per gpu (default: 1024)')
    group.add_argument('--model_kwargs', nargs='*', default={}, action=utils.ParseKwargs)

    # Misc
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--log_interval', type=int, default=10,
                       help='Log every x batches during validation procedure')
    group.add_argument('--seed', type=int, default=0, metavar='S',
                       help='random seed (default: 0)')
    group.add_argument('--worker_seeding', type=str, default='all',
                       help='worker seed mode (default: all)')
    group.add_argument('--workers', type=int, default=8, metavar='N',
                       help='how many training processes to use (default: 8)')
    group.add_argument('--pin_mem', action='store_true',
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no_pin_mem', action='store_true', dest='pin_mem',
                       help='Disable pinning CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.set_defaults(pin_mem=True)
    group.add_argument('--no_prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher -- Prefetcher uses async cuda streams to speed up loading.')
    group.add_argument('--eval_metric', default='acc1', type=str, metavar='EVAL_METRIC',
                       help='Best metric (default: "acc1"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local_rank", default=0, type=int, help='Leave untouched. Internal use.')

    # Experiment tracking arguments
    group.add_argument('--log_wandb', action='store_true',
                       help='log training and validation metrics to wandb')
    group.add_argument('--wandb_projname', default='BiXT_evals', type=str,
                       help='project used for logging progress to wandb')

    return parser


def _parse_args():
    parser = argparse.ArgumentParser('BiXT evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    return args


def main():
    utils.setup_default_logging()

    # ========== Obtain experiment settings from arg-parser ==========
    args = _parse_args()
    args.prefetcher = not args.no_prefetcher

    # ========== Set up cuda, initialise distributed training if multiple GPUs are used ==========
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # Set seed for 'partial' reproducibility -- but we'll go through the entire val data anyways
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
        _logger.info(f"Creating model: {args.model} and loading weights from {args.model_checkpoint}.")
    if 'bixt' in args.model:  # Only support BiXT models for now -- can easily be extended to others if desired
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=0.,
            sa_drop_path_rate=0.,
            ca_drop_path_rate=0.,
            drop_block_rate=None,
            checkpoint_path=args.model_checkpoint
        )
    else:
        raise NotImplementedError("Model currently not supported.")

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} successfully created and weights loaded, '
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

    # setup distributed training
    if args.distributed:
        model = NativeDDP(model, device_ids=[device])

    #
    # ========== Preparing the dataset and augmentation procedures ==========
    #
    # FIXME: Note -> This only really sets defaults for imagenet, not for other datasets! -> FIX if needed
    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # create the train and eval datasets
    if utils.is_primary(args):
        _logger.info("Creating data set...")
    ds_name = 'torch/image_folder' if args.dataset == 'imagenet' else args.dataset

    dataset_val = create_dataset(
        ds_name,
        root=args.data_path,
        split=args.val_split,
        is_training=False,
        download=args.dataset_download,
        batch_size=args.validation_batch_size,
    )

    if utils.is_primary(args):
        _logger.info(f"Using the {args.dataset} dataset to run evaluation.\n"
                     f"Data successfully loaded: There are {len(dataset_val)} validation images available.")

    # create data loaders
    if utils.is_primary(args):
        _logger.info("Creating data loader...")

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # Reduces validation padding issues when using TFDS, WDS w/ workers and distributed
        eval_workers = min(2, args.workers)

    loader_eval = create_loader(
        dataset_val,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size,  # * 1.5),
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

    # == Set up loss function ==
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # == Set up optional wandb logging of results
    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.wandb_projname, entity="your_username", config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`")

    #
    # ========== Start the validation procedure ==========
    #
    try:
        # Run validation on the validation set and obtain metrics
        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            args
        )

        # Log the evaluation metrics to wandb, if desired (validation loss, acc@1 and acc@5)
        if utils.is_primary(args) and args.log_wandb and has_wandb:
            wandb.log(eval_metrics)

    except KeyboardInterrupt:
        pass

    if utils.is_primary(args):
        print("\n### >>>> SUMMARY <<< ###")
        print(f"### > Validation Accuracy Top1: {eval_metrics['acc1']:.3f} "
              f"| Validation Accuracy Top5: {eval_metrics['acc5']:.3f} "
              f"| Validation Loss: {eval_metrics['loss']:.3f}")
        print('### Done. ###')


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

    print("\n>> We hope you had fun evaluating your BiXT model(s)!")
    print(">> If you like our work, please cite our paper 'https://arxiv.org/pdf/2402.12138'\n"
          ">> and consider giving us a star on github 'https://github.com/mrkshllr/BiXT'.")