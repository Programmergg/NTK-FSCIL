import torch
import random
import numpy as np
import torchvision.transforms as transforms
from dataloader.RandAugment import augmix_ops, AugMixAugment

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(img1, img2, lam):
    bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)
    mixed_img = img1.clone()
    mixed_img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
    return mixed_img

augmix_transformer = AugMixAugment(ops=augmix_ops())

def combine_with_augmix(img1, img2, lam):
    # Convert tensors to PIL Images
    img1_pil = transforms.ToPILImage()(img1)
    img2_pil = transforms.ToPILImage()(img2)
    # Apply AugMix on both images
    img1_aug = augmix_transformer(img1_pil)
    img2_aug = augmix_transformer(img2_pil)
    # Convert back to tensor
    img1_aug_tensor = transforms.ToTensor()(img1_aug)
    img2_aug_tensor = transforms.ToTensor()(img2_aug)
    combined_img = lam * img1_aug_tensor + (1 - lam) * img2_aug_tensor
    return combined_img.cuda()

def split_into_blocks(img, block_num):
    C, H, W = img.size()
    block_H, block_W = H // block_num, W // block_num
    blocks = []
    for i in range(block_num):
        for j in range(block_num):
            block = img[:, i * block_H:(i + 1) * block_H, j * block_W:(j + 1) * block_W]
            blocks.append(block)
    return blocks

def combine_blocks(blocks, block_num):
    rows = []
    for i in range(block_num):
        row_blocks = blocks[i * block_num:(i + 1) * block_num]
        rows.append(torch.cat(row_blocks, dim=2))
    return torch.cat(rows, dim=1)

def puzzle_mix(img1, img2, block_num):
    blocks1 = split_into_blocks(img1, block_num)
    blocks2 = split_into_blocks(img2, block_num)
    mixed_blocks = []
    for block1, block2 in zip(blocks1, blocks2):
        if random.random() > 0.5:
            mixed_blocks.append(block1)
        else:
            mixed_blocks.append(block2)
    mixed_img = combine_blocks(mixed_blocks, block_num)
    return mixed_img