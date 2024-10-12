import os
import torch
import numpy as np
import os.path as osp
from PIL import Image
from torchvision import transforms
from dataloader.transforms import *
from torch.utils.data import Dataset

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias
        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
        img = self.tensor_to_pil(img)
        return img

class RandomTranslateWithReflect:
    '''
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    '''
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation, self.max_translation + 1, size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size
        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)
        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))
        new_image.paste(old_image, (xpad, ypad))
        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))
        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))
        new_image = new_image.crop((xpad - xtranslation, ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image

class ImageNet(Dataset):
    def __init__(self, root, args, train=True, index_path=None, index=None, base_sess=None, validation=False, augment=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'ImageNet100/images')
        self.SPLIT_PATH = os.path.join(root, 'ImageNet100/split')
        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1
        self.wnids = []
        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        self.args = args
        self.repeat = self.args.train_shot + self.args.train_query
        self.image_size = 112
        self.augment = augment
        if train:
            if validation:
                print('---- ImageNet Train Transform for Validation ---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            elif not base_sess:
                print('---- ImageNet Train Transform for Incremental Classes ---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            else:
                print('---- ImageNet OneCrops Training Transform ---')
                self.transform = self.get_transform(self.augment, self.image_size, 'train')
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            if validation:
                print('---- ImageNet Testing Transform for Validation ---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            else:
                print('---- ImageNet Testing Transform for Testing ---')
                self.transform = self.get_transform('test_aug', self.image_size, 'test')
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def get_transform(self, augment, image_size, setname):
        if setname == 'train':
            if augment == 'AMDIM':
                transforms_list = self.AMDIM_transforms(image_size)
            elif augment == 'SimCLR':
                transforms_list = self.SimCLR_transforms(image_size)
            elif augment == 'AutoAug':
                transforms_list = self.AutoAug_transforms(image_size)
            elif augment == 'RandAug':
                transforms_list = self.RandAug_transforms(image_size)
            elif augment == 'Normal':
                transforms_list = [
                    transforms.Resize([128, 128]),
                    transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]
            elif augment == 'validation_aug':
                transforms_list = self.test_transforms(image_size)
            else:
                raise ValueError(
                    f'Non-supported Augmentation Type: {augment}. Please Revise Data Pre-Processing Scripts.')
        else:
            transforms_list = self.test_transforms(image_size)
        transform = transforms.Compose(
            transforms_list + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return transform

    def test_transforms(self, image_size):
        transforms_list = [
            transforms.Resize([128, 128]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AMDIM_transforms(self, image_size):
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        transforms_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.25),
            transforms.ToTensor(),
        ]
        return transforms_list

    def SimCLR_transforms(self, image_size):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        transforms_list = [
            transforms.RandomResizedCrop(size=image_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def AutoAug_transforms(self, image_size):
        from dataloader.autoaug import RandAugment
        transforms_list = [
            RandAugment(2, 12),
            ERandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8),
            transforms.ToTensor(),
            Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
        ]
        return transforms_list

    def RandAug_transforms(self, image_size):
        from dataloader.RandAugment import rand_augment_transform
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transforms_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomApply([GaussianBlur(22)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
        return transforms_list

    def SelectfromTxt(self, data2label, index_path):
        index = []
        for i in range(len(index_path)):
            lines = [x.strip() for x in open(index_path[i], 'r').readlines()]
            for line in lines:
                index.append(line)
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets