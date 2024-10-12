import os
import re
import sys
import torch
import errno
import shutil
import hashlib
import tempfile
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from urllib.request import urlopen
from urllib.parse import urlparse  # noqa: F401
from models.backbones.criterion import CurricularFacePenaltySMLoss, AngularPenaltySMLoss

def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'.format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def _get_torch_home():
    torch_home = os.path.expanduser(os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home

__all__ = ['ResNet','resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')
    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim)),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            # nn.utils.spectral_norm(nn.Linear(hidden_dim, out_dim)),
            nn.BatchNorm1d(out_dim, affine=False)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(in_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.01),
        #     nn.Linear(hidden_dim, out_dim),
        #     nn.BatchNorm1d(out_dim, affine=False)
        # )

    def forward(self, x):
        x = self.fc(x)
        return x

def record_grad_norm(module):
    def hook(grad):
        norms = torch.norm(grad, dim=1)
        norm = torch.sum(norms)
        module.grad_norm_accumulator.add(norm.item(), norms.size(0))
    return hook

class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x, n=1):
        self.v = (self.v * self.n + x) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = Encoder.get_backbone(self.args)
        out_dim = self.backbone.out_dim
        if self.args.dataset == 'cub200':
            print('Model | load pre-trained model.')
            model_dict = self.backbone.state_dict()
            state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
            state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        self.projector = projection_MLP(out_dim, self.args.feat_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)

        total_num_of_cls = self.args.num_classes
        if args.data_fusion:
            aug_for_base = int(((args.base_class) * (args.base_class - 1))/2)
            aug_for_inc = int(((args.way) * (args.way - 1))/2)
            # print('Model | aug_for_base: {0}, aug_for_inc: {1}'.format(aug_for_base, aug_for_inc))
            aug_num_of_cls = aug_for_base + (args.sessions - 1) * aug_for_inc
            total_num_of_cls = total_num_of_cls + aug_num_of_cls

        if self.args.loss_type == 'curriculum':
            self.cls_loss_fn = CurricularFacePenaltySMLoss(s=self.args.loss_s, m=self.args.loss_m)
        else:
            self.cls_loss_fn = AngularPenaltySMLoss(loss_type=self.args.loss_type, s=self.args.loss_s, m=self.args.loss_m)
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(self.args.feat_dim, total_num_of_cls, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.grad_norm_accumulator = Averager()
        self.emb_norm_accumulator = Averager()

    @staticmethod
    def get_backbone(args):
        backbone_mapping = {
            'resnet18': {
                'cifar100': ('models.backbones.resnet18_width', 'ResNet18', {'width': args.resnet_width}),
                'miniimagenet': ('models.backbones.resnet18_width', 'ResNet18', {'width': args.resnet_width}),
                'cub200': ('models.backbones.resnet_cub', 'ResNet18', {}),
                'imagenet100': ('models.backbones.resnet18_width', 'ResNet18', {'width': args.resnet_width}),
            },
            'resnet12': {
                'cifar100': ('models.backbones.resnet12_width', 'ResNet12', {'width': args.resnet_width}),
                'miniimagenet': ('models.backbones.resnet12_width', 'ResNet12', {'width': args.resnet_width}),
                'imagenet100': ('models.backbones.resnet12_width', 'ResNet12', {'width': args.resnet_width}),
            },
            'vit_tiny': {
                'cifar100': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'miniimagenet': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'cub200': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'imagenet100': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
            },
            'vit_small': {
                'cifar100': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'miniimagenet': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'cub200': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'imagenet100': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
            }
        }
        if args.network_type in backbone_mapping:
            if args.dataset in backbone_mapping[args.network_type]:
                module_name, class_name, kwargs = backbone_mapping[args.network_type][args.dataset]
                ModuleClass = getattr(__import__(module_name, fromlist=[class_name]), class_name)
                return ModuleClass(**kwargs)
        raise RuntimeError('Something is wrong.')

    def split_instances_normal(self, num_shot, num_query, num_way, num_class=None):
        num_class = num_way if (num_class is None or num_class < num_way) else num_class
        permuted_ids = torch.zeros(num_shot + num_query, num_way).long()
        # select class indices
        clsmap = torch.randperm(num_class)[:num_way]
        # ger permuted indices
        for j, clsid in enumerate(clsmap):
            permuted_ids[:, j].copy_(torch.randperm((num_shot + num_query)) * num_class + clsid)
        if torch.cuda.is_available():
            permuted_ids = permuted_ids.cuda()
        support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=0)
        return support_idx, query_idx

    def split_instances(self):
        return self.split_instances_normal(self.args.train_shot, self.args.train_query, self.args.train_way)

    def meta_forward(self, x, **kwargs):
        # feature extraction
        x = x.squeeze(0)
        instance_embs = self.encoder(x)
        # split support query set for few-shot data
        support_idx, query_idx = self.split_instances()
        # print(instance_embs.shape, support_idx.shape, query_idx.shape)
        # torch.Size([360, 2048]) torch.Size([1, 60]) torch.Size([5, 60])
        logits = self._forward(instance_embs, support_idx, query_idx, **kwargs)
        instance_embs.register_hook(record_grad_norm(self))
        norms = torch.norm(instance_embs, dim=1)
        norm = torch.sum(norms)
        self.emb_norm_accumulator.add(norm.item(), norms.size(0))
        label = torch.arange(self.args.train_way, dtype=torch.long).repeat(self.args.train_query).cuda()
        return logits, label

    def forward(self, x):
        x = self.encoder(x)
        x = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.classifier.weight, p=2, dim=1))
        return x

    def _forward(self, instance_embs, support_idx, query_idx, **kwargs):
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,))).unsqueeze(0)
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,))).unsqueeze(0)
        # print(support.shape, query.shape) # torch.Size([1, 1, 60, 2048]) torch.Size([1, 5, 60, 2048])
        return self._forward_task(support, query, **kwargs)

    def _forward_task(self, support, query, **kwargs):
        # get mean of the support
        proto = support.mean(dim=1)
        # query: (num_batch, num_query, num_way, num_emb)
        # proto: (num_batch, num_way, num_emb)
        if self.args.similarity == 'euclidean':
            logits = self.euclidean(proto, query)
        elif self.args.similarity == 'cosine': # cosine similarity: more memory efficient
            logits = self.cosine(proto, query)
        elif self.args.similarity == 'mahalanobis':
            logits = self.Mahalanobis(proto, query)
        return logits

    def cosine(self, proto, query):
        num_proto = proto.shape[1]
        proto = F.normalize(proto, dim=-1)  # normalize for cosine similarity
        query = F.normalize(query, dim=-1)
        logits = torch.einsum('ijk,ilmk->ilmj', proto, query) / self.args.temperature
        # proto = proto.view(proto.size(0), 1, 1, *proto.shape[1:])
        # query = query.view(*query.shape[:3], 1, query.size(3))
        # logits = F.cosine_similarity(proto, query, dim=-1)
        logits = logits.reshape(-1, num_proto)
        return logits

    def euclidean(self, proto, query):
        emb_dim = proto.size(-1)
        num_query = np.prod(query.shape[1:3])
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)
        logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        return logits

    def Mahalanobis(self, proto, query):
        emb_dim = proto.size(-1)
        num_query = np.prod(query.shape[1:3])
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)
        dif = proto - query
        logits = - torch.einsum('ijk,kl,ijl->ij', dif, self.mat, dif) / self.args.temperature
        return logits