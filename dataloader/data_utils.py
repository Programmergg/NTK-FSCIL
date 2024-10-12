import torch
import numpy as np
from dataloader.sampler import CategoriesSampler

def examplar_collate(batch):
    X, Y = [], []
    for b in batch:
        X.append(torch.stack(b[0]))
        Y.append(b[1])
    X = torch.stack(X)
    label = torch.LongTensor(Y)
    img = torch.cat(tuple(X.permute(1, 0, 2, 3, 4)), dim=0)
    # (repeat * class , *dim)
    return img, label

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        from .cifar100 import cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        from .cub200 import cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'miniimagenet':
        from .miniimagenet import miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'imagenet100':
        from .imagenet100 import imagenet100 as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    args.Dataset = Dataset
    return args

def get_train_dataloader(args, session):
    if session == 0:
        trainset, trainloader = get_base_train_dataloader(args)
    else:
        trainset, trainloader = get_new_dataloader(args, session)
    return trainset, trainloader

def get_base_train_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=True, download=True, index=class_index, base_sess=True, augment=args.augment)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, augment=args.augment)
    if args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, augment=args.augment)
    if args.dataset == 'imagenet100':
        trainset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, augment=args.augment)
    if args.dataset != 'imagenet100':
        batch_sampler = CategoriesSampler(trainset.targets, args.epochs_base, args.train_way, args.train_shot + args.train_query)
    else:
        batch_sampler = CategoriesSampler(trainset.targets, args.epochs_base * 20, args.train_way, args.train_shot + args.train_query)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.epochs_base, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader

def get_new_dataloader(args, session):
    txt_path_list = []
    txt_path = "./data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    txt_path_list.append(txt_path)
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=True, download=False, index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False)
    if args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False)
    if args.dataset == 'imagenet100':
        trainset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True, num_workers=args.num_workers, drop_last=True)
    return trainset, trainloader

def get_incremental_dataset_fs(args, session=None):
    class_index = []
    if session == None:
        session = args.sessions
    print('session: {0}'.format(session))
    txt_path_list = []
    for i in range(session + 1):
        if i == 0:
            txt_path = "./data/index_list/" + args.dataset + '/session_{0}'.format(str(i + 1)) + '.txt'
        else:
            txt_path = "./data/index_list/" + args.dataset + '/session_{0}'.format(str(i + 1)) + '.txt'
        temp_class_index = open(txt_path).read().splitlines()
        for j in range(len(temp_class_index)):
            class_index.append(temp_class_index[j])
        txt_path_list.append(txt_path)
    print('number of images: {0}'.format(len(class_index)))
    print('~~~~~~~~ training dataset ~~~~~~~~')
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=True, download=True, index=class_index, base_sess=False, validation=True)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False, validation=True)
    if args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False, validation=True)
    if args.dataset == 'imagenet100':
        trainset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=True, index_path=txt_path_list, base_sess=False, validation=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_base, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print('~~~~~~~~ testing dataset ~~~~~~~~')
    class_new = get_session_classes(args, session)  # test on all encountered classes
    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=False, download=False, index=class_new, base_sess=False, validation=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, args=args, train=False, index=class_new, base_sess=False, validation=False)
    if args.dataset == 'miniimagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=False, index=class_new, base_sess=False, validation=False)
    if args.dataset == 'imagenet100':
        testset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=False, index=class_new, base_sess=False, validation=False)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testset, testloader

def get_validation_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=True, download=True, index=class_index, base_sess=True, validation=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, args=args, train=False, download=False, index=class_index, base_sess=False, validation=True)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, validation=True)
        testset = args.Dataset.CUB200(root=args.dataroot, args=args, train=False, index=class_index, base_sess=False, validation=True)
    if args.dataset == 'miniimagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, validation=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, args=args, train=False, index=class_index, base_sess=False, validation=True)
    if args.dataset == 'imagenet100':
        trainset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=True, index=class_index, base_sess=True, validation=True)
        testset = args.Dataset.ImageNet(root=args.dataroot, args=args, train=False, index=class_index, base_sess=False, validation=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_base, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testset, testloader

def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list