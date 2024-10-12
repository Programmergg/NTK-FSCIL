import os
import torch
import pickle
import os.path
import numpy as np
from PIL import Image
from dataloader.transforms import *
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

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

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(self, root, args, train=True, download=False, index=None, base_sess=None, validation=False, augment=None):
        super(CIFAR10, self).__init__(root)
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')
        self.args = args
        self.repeat = self.args.train_shot + self.args.train_query
        self.image_size = 32
        self.augment = augment
        if self.train:
            downloaded_list = self.train_list
            if validation:
                print('---- CIFAR100 Train Transform for Validation---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            elif not base_sess:
                print('---- CIFAR100 Train Transform for Incremental Classes---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            else:
                print('---- CIFAR100 OneCrops Training Transform ---')
                self.transform = self.get_transform(self.augment, self.image_size, 'train')
        else:
            downloaded_list = self.test_list
            if validation:
                print('---- CIFAR100 Testing Transform for Validation---')
                self.transform = self.get_transform('validation_aug', self.image_size, 'train')
            else:
                print('---- CIFAR100 Testing Transform for Testing---')
                self.transform = self.get_transform('test_aug', self.image_size, 'test')

        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.asarray(self.targets)
        if base_sess:
            self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        else:
            if train:
                self.data, self.targets = self.NewClassSelector(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromDefault(self.data, self.targets, index)
        self._load_meta()

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
                    transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                ]
            elif augment == 'validation_aug':
                transforms_list = [transforms.ToTensor(),]
            else:
                raise ValueError(f'Non-supported Augmentation Type: {augment}. Please Revise Data Pre-Processing Scripts.')
        else:
            transforms_list = self.test_transforms(image_size)
        transform = transforms.Compose(transforms_list + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return transform

    def test_transforms(self, image_size):
        if image_size == 32:
            resize = int((8 / 7) * image_size)
        else:
            resize = int((92 / 84) * image_size)
        transforms_list = [
            transforms.Resize(resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
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

    def SelectfromDefault(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        return data_tmp, targets_tmp

    # def NewClassSelector(self, data, targets, index, rotate=False, fusion=False):
    #     data_tmp = []
    #     targets_tmp = []
    #     ind_list = [int(i) for i in index]
    #     ind_np = np.array(ind_list)
    #     # index = ind_np.reshape((5,5))
    #     if len(ind_list) % 5 == 0:
    #         category = int(len(ind_list) / 5)
    #         index = ind_np.reshape((category, 5))
    #     else:
    #         raise RuntimeError('Something is wrong.')
    #     for i in index:
    #         ind_cl = i
    #         if data_tmp == []:
    #             data_tmp = data[ind_cl]
    #             targets_tmp = targets[ind_cl]
    #         else:
    #             data_tmp = np.vstack((data_tmp, data[ind_cl]))
    #             targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
    #     return data_tmp, targets_tmp
    def NewClassSelector(self, data, targets, index, rotate=False, fusion=False):
        data_tmp = None
        targets_tmp = None
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list)
        if len(ind_list) % 5 == 0:
            category = int(len(ind_list) / 5)
            index = ind_np.reshape((category, 5))
        else:
            raise RuntimeError('Something is wrong.')
        for i in index:
            ind_cl = i
            if data_tmp is None:  # Initializing the first batch of data
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        return data_tmp, targets_tmp

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        data, label = self.data[index], self.targets[index]
        img = Image.fromarray(data)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }