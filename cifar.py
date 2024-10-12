import argparse
import importlib
from utils import *

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and network
    parser.add_argument('-project', type=str, default='base')
    parser.add_argument('-dataset', type=str, default='cifar100', choices=['miniimagenet', 'cub200', 'cifar100', 'imagenet100', 'imagenet1000'])
    parser.add_argument('-dataroot', type=str, default='data/')
    # about pre-training
    parser.add_argument('--augment', type=str, default='Normal', choices=['Normal', 'AMDIM', 'SimCLR', 'AutoAug', 'RandAug'])
    parser.add_argument('--network-type', type=str, default='resnet18', choices=['resnet12', 'resnet18', 'vit_tiny', 'vit_small'])
    parser.add_argument('--loss-type', type=str, default='curriculum', choices=['curriculum', 'arcface', 'sphereface', 'cosface', 'crossentropy'])
    parser.add_argument('--resnet_width', type=int, default=128)
    parser.add_argument('--pretrain_weights', type=str, default='spark', choices=['dino', 'spark', 'mae', 'moco-v3', 'simclr', 'byol'])
    parser.add_argument('-epochs_base', type=int, default=300)
    parser.add_argument('-lr_base', type=float, default=0.01)
    parser.add_argument('-schedule', type=str, default='Milestone', choices=['Step', 'Milestone', 'Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-batch_size_base', type=int, default=128, help='set this for validation, not for training')
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-regularization_temp', type=float, default=0.005)
    parser.add_argument('-save_path', type=str, default='./checkpoint')
    parser.add_argument('-resume', type=str, default='')
    parser.add_argument('--save_ntk', default=False, help='whether to retain ntk properties')
    # about training parameters
    parser.add_argument('-gpu', default='1')
    parser.add_argument('-num_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument('--loss_s', type=float, default=25.0)
    parser.add_argument('--loss_m', type=float, default=0.1)
    parser.add_argument('--meta_temp', type=float, default=0.3)
    parser.add_argument('--data_fusion', default=True, help='Fuse the input data to increase the categories.')
    # NTK meta-learning parameters
    parser.add_argument('--train_way', type=int, default=30)
    parser.add_argument('--train_shot', type=int, default=1)
    parser.add_argument('--train_query', type=int, default=3)
    parser.add_argument('--similarity', type=str, default='cosine', choices=['mahalanobis', 'cosine', 'euclidean'])
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser

if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer_alice' % (args.project)).FSCILTrainer(args)
    trainer.train()