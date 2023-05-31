import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from models import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rand_bbox_out(size, lam):
    s0 = size[0]
    s1 = size[1]
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mask = np.ones((s0, s1, W, H), np.float32)
    mask[:, :, bbx1: bbx2, bby1: bby2] = 0.
    mask = torch.from_numpy(mask)
    return mask

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def cutout_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # y_a, y_b = y, y[index]
    mask = rand_bbox_out(x.size(), lam)

    mask = mask.to(device)
    x_cutout = x * mask
    # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x_cutout, y

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam

def pic_show(a,b,c):
    pic_a = a.cpu().numpy()
    pic_b = b.cpu().numpy()
    pic_c = c.cpu().numpy()

    pic_a = np.expand_dims(pic_a,0)
    pic_b = np.expand_dims(pic_b, 0)
    pic_c = np.expand_dims(pic_c, 0)



    transformed_images = np.concatenate((pic_a,pic_b,pic_c),0)

    # 标签列表
    labels = [["cutout_pic_1", "cutout_pic_2", "cutout_pic_3"],
              ["cutmix_pic_1", "cutmix_pic_2", "cutmix_pic_3"],
              ["mixup_pic_1", "mixup_pic_2", "mixup_pic_3",]]

    # 创建一个3x3的子图布局
    fig, axes = plt.subplots(3, 3, figsize=(10, 20))

    # 遍历每个子图，显示原图像和变换图像，并添加标签
    for i in range(3):
        # 显示原图像
        axes[i, 0].imshow(np.transpose(transformed_images[i][0], (1, 2, 0)).astype(np.uint8))
        axes[i, 0].set_title(labels[i][0],size = 5)

        axes[i, 1].imshow(np.transpose(transformed_images[i][1], (1, 2, 0)).astype(np.uint8))
        axes[i, 1].set_title(labels[i][1],size = 5)

        axes[i, 2].imshow(np.transpose(transformed_images[i][2], (1, 2, 0)).astype(np.uint8))
        axes[i, 2].set_title(labels[i][2],size = 5)

    # 显示图像
    plt.show()

print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=3, shuffle=True, num_workers=0)



for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    inputs_cout, _ = cutout_data(inputs,targets)
    inputs_cmix, _,_,_ = cutmix_data(inputs, targets)
    inputs_mix, _,_,_ = mixup_data(inputs,targets)

    pic_show(inputs_cout,inputs_cmix,inputs_mix)
    break


