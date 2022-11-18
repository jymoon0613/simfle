import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
import random
import os
import numpy as np
from PIL import Image

class SimFLEDataset(Dataset):
    def __init__(self, root, img_size_G=224, img_size_F=112):

        self.files = sorted(glob.glob("%s/*.jpg" % root))
        self.normalize = transforms.Normalize(mean = [0.5795, 0.4522, 0.3957], std = [0.2769, 0.2473, 0.2412])
        self.augmentations = get_augmentations(size=img_size_G, normalize=self.normalize)
        self.GFL_transform = MultiViewTransform([self.augmentations, self.augmentations])
        self.FFL_transform = transforms.Compose([
            transforms.Resize(img_size_F),
            transforms.ToTensor(),
            self.normalize])

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img_GFL = self.GFL_transform(img)
        img_FFL = self.FFL_transform(img)

        return img_FFL, img_GFL

    def __len__(self):
        return len(self.files)

class GaussianBlur(object):

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
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

class MultiViewTransform(object):
    def __init__(self, *args):
        self.transforms = args[0]
        self.random_flip = transforms.RandomHorizontalFlip()

    def __call__(self, sample, *with_consistent_flipping):
        if with_consistent_flipping:
            sample = self.random_flip(sample)
        output = [transform(sample) for transform in self.transforms]

        return output

def get_augmentations(size, normalize, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    # SimCLR : https://arxiv.org/abs/2002.05709
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    augmentations = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * size)),
                                          transforms.ToTensor(),
                                          normalize])

    return augmentations
