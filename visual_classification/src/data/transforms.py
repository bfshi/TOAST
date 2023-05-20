#!/usr/bin/env python3

"""Image transformations."""
import torchvision as tv
import numpy as np
from .imagenet_c import corrupt

class CorruptionTransform(object):
    def __init__(self, corruption_name, severity):
        self.corruption_name = corruption_name
        self.severity = severity

    def __call__(self, img):
        if self.severity >= 1e-4:
            return corrupt(np.array(img), severity=self.severity, corruption_name = self.corruption_name)
        else:
            return img

def get_transforms(split, size, cfg):
    normalize = tv.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if size == 448:
        resize_dim = 512
        crop_dim = 448
    elif size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384
    if split == "train":
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                # tv.transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                # tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                CorruptionTransform(cfg.DATA.CORRUPTION, cfg.DATA.SEVERITY),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    return transform
