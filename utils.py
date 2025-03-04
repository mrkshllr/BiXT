""" Helpers for argument handling as well as dataset creation

Files are part of the official implementation of the paper:
'Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers'
-> Published @ NeurIPS 2024, or see https://arxiv.org/pdf/2402.12138

Arguments saved to yaml for easier tracking / interpretation of experiments later on;
Hashlib used to create unique hashes for checkpoint storage to avoid overwriting different experiments

"""
import yaml
import hashlib


def _args_to_yaml(args):
    return yaml.safe_dump(args.__dict__, default_flow_style=False)


def _get_num_classes(dataset_name):
    # Returning the number of classes for popular datasets
    if 'cifar100' in dataset_name:
        return 100
    elif 'cifar10' in dataset_name:
        return 10
    elif 'imagenet' or 'image_folder' in dataset_name:
        return 1000
    else:
        return None


def _get_default_datasize(dataset_name):
    # Returning the default image size for popular datasets
    if 'cifar' in dataset_name:
        return (3, 32, 32)
    elif 'imagenet' or 'image_folder' in dataset_name:
        return (3, 224, 224)
    else:
        return None


def _get_hash_from_args(args, non_essential_keys=None):
    """ Returns a unique hash for an argparse object. Also takes additional keys that are non-essential for the hash
    and should be removed, i.e. keys that might change but still represent the same run (dataset path, etc.) """
    args_cpy = args.__dict__.copy()
    if non_essential_keys is not None:
        for k in non_essential_keys:
            try:
                args_cpy.pop(k)
            except BaseException:
                print(f"Key{k} not contained in namespace! Consider removing from list of non-essential keys.")

    args_cpy = str(args_cpy)
    arghash = hashlib.md5(args_cpy.encode()).hexdigest()
    return arghash

