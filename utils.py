import os
import time
import torch
import random
import numpy as np
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()

def load_dino_pretrained_weights(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # remove prefix
    for key in state_dict.keys():
        if 'classifier' in key:
            del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

def load_self_pretrain_weights(args, model):
    weights_paths = {
        'cifar100': {
            'dino': {
                'resnet18': 'pretrain_weights/cifiar/dino/resnet18/checkpoint.pth',
                'vit_tiny': 'pretrain_weights/cifiar/dino/vit_tiny/checkpoint.pth',
                'vit_small': 'pretrain_weights/cifiar/dino/vit_small/checkpoint.pth',
            },
            'spark': {
                'resnet18': 'pretrain_weights/cifiar/spark/resnet18/checkpoint.pth',
            },
            'mae': {
                'vit_tiny': 'pretrain_weights/cifiar/mae/vit_tiny/checkpoint.pth',
                'vit_small': 'pretrain_weights/cifiar/mae/vit_small/checkpoint.pth',
            },
            'moco-v3': {
                'resnet18': 'pretrain_weights/cifiar/moco-v3/resnet18/checkpoint.pth',
                'vit_tiny': 'pretrain_weights/cifiar/moco-v3/vit_tiny/checkpoint.pth',
                'vit_small': 'pretrain_weights/cifiar/moco-v3/vit_small/checkpoint.pth',
            },
            'simclr': {
                'resnet18': 'pretrain_weights/cifiar/simclr/resnet18/checkpoint.pth',
                'vit_tiny': 'pretrain_weights/cifiar/simclr/vit_tiny/checkpoint.pth',
                'vit_small': 'pretrain_weights/cifiar/simclr/vit_small/checkpoint.pth',
            },
            'byol': {
                'resnet18': 'pretrain_weights/cifiar/byol/resnet18/checkpoint.pth',
                'vit_tiny': 'pretrain_weights/cifiar/byol/vit_tiny/checkpoint.pth',
                'vit_small': 'pretrain_weights/cifiar/byol/vit_small/checkpoint.pth',
            },
        },
        'miniimagenet': {
            'spark': {
                'resnet18': 'pretrain_weights/miniimagenet/spark/resnet18/checkpoint.pth',
                'resnet12': 'pretrain_weights/miniimagenet/spark/resnet12/checkpoint.pth',
            }
        },
        'imagenet100': {
            'spark': {
                'resnet18': 'pretrain_weights/imagenet100/spark/resnet18/checkpoint.pth',
            }
        }
    }
    resume_from = weights_paths.get(args.dataset, {}).get(args.pretrain_weights, {}).get(args.network_type, '')
    if not resume_from:
        print('Please specify self-supervised pre-training weights!!!!!!!')
        return
    if args.pretrain_weights == 'dino':
        # load_dino_pretrained_weights(model.backbone, resume_from, 'teacher')
        state_dict = torch.load(resume_from, map_location="cpu")['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove prefix
        for key in state_dict.keys():
            if 'classifier' in key:
                del state_dict[key]
        msg = model.backbone.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(resume_from, msg))
    else:
        state_dict = torch.load(resume_from, map_location="cpu")
        if args.pretrain_weights in ['spark', 'simclr']:
            state_dict = state_dict.get('module', state_dict)
        elif args.pretrain_weights == 'mae':
            state_dict = state_dict['model']
            state_dict = {key: value for key, value in state_dict.items() if 'decoder' not in key}
        elif args.pretrain_weights == 'byol':
            state_dict = state_dict['base_model']
        elif args.pretrain_weights == 'moco-v3':
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.base_encoder'):
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                del state_dict[k]
        missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
        print(f'[load_checkpoint] missing_keys={missing}')
        print(f'[load_checkpoint] unexpected_keys={unexpected}')

def pprint(x):
    _utils_pp.pprint(x)

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))
    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")
    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=True)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))
    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def set_seed(seed):
    if seed is not None:
        print('static seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        seed = random.randint(1, 10000)
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)