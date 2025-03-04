""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def update_summary_ours(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
        num_params=None,
        wd=None,
        test_metrics=None,
):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if test_metrics:  # For LRA experiments -- only validation acc is used to pick best model! (but S5 paper logs all)
        rowd.update([('val_' + k, v) for k, v in eval_metrics.items()])
        rowd.update([('test_' + k, v) for k, v in test_metrics.items()])
    else:  # Default ImageNet logging
        rowd.update([('test_' + k, v) for k, v in eval_metrics.items()])
    if num_params is not None:
        rowd['n_parameters'] = num_params
    if lr is not None:
        rowd['train_lr'] = lr
    if wd is not None:
        rowd['weight_decay'] = wd
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)


def update_summary_ours_steps(
        step,
        train_metrics,
        eval_metrics,
        filename,
        lr=None,
        write_header=False,
        log_wandb=False,
        num_params=None,
        wd=None,
        test_metrics=None,
):
    rowd = OrderedDict(step=step)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if test_metrics:  # For LRA experiments -- only validation acc is used to pick best model! (but S5 paper logs all)
        rowd.update([('val_' + k, v) for k, v in eval_metrics.items()])
        rowd.update([('test_' + k, v) for k, v in test_metrics.items()])
    else:  # Default ImageNet logging
        rowd.update([('test_' + k, v) for k, v in eval_metrics.items()])
    if num_params is not None:
        rowd['n_parameters'] = num_params
    if lr is not None:
        rowd['train_lr'] = lr
    if wd is not None:
        rowd['weight_decay'] = wd
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
