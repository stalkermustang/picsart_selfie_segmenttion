import albumentations

from albumentations import torch as AT

train_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.HorizontalFlip(),
    albumentations.Rotate(limit=10),
    albumentations.JpegCompression(80),
    albumentations.HueSaturationValue(),
    albumentations.Normalize(),
    AT.ToTensor()
])

val_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.Normalize(),
    AT.ToTensor()
])

test_norm_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.Normalize(),
    AT.ToTensor()
])

test_flip_transforms = albumentations.Compose([
    albumentations.Resize(256, 256),
    albumentations.HorizontalFlip(p=1.1),
    albumentations.Normalize(),
    AT.ToTensor()
])
