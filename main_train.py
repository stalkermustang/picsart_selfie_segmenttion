import argparse
import os

import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import train_transforms, val_transforms
from dataset import FaceDataset
from loss import Loss
from nets.net_se_resnext101 import MyNetSeResnext101
from nets.net_se_resnext50 import MyNetSeResnext50
from nets.net_senet154 import MyNetSeNet154
from utils import cuda, get_split, save_model

from config import TRAIN_DATA_DIR, TEST_DATA_DIR, TRAIN_MASK_DIR, CHECKPOINT_DIR

BS = 24


model_list = {
    'se_resnext101': MyNetSeResnext101,
    'senet154': MyNetSeNet154,
    'se_resnext50': MyNetSeResnext50,
}


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device_ids', type=str, default='0',
        help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--lr', type=float, default=0.0001)
    arg('--num_epoch', type=int, default=40)
    arg('--batch_size', type=int, default=BS)
    arg('--model', type=str, default='se_resnext101',
        choices=list(model_list.keys()))

    args = parser.parse_args()
    GPUs = [int(i) for i in args.device_ids.split(',')]

    path_images = np.array(
        list(filter(lambda x: x.endswith('.jpg'), os.listdir(TRAIN_DATA_DIR))))

    train_images, val_images = get_split(path_images, args.fold)

    train_dataset = FaceDataset(
        images_dir=TRAIN_DATA_DIR,
        images_name=train_images,
        target_dir=TRAIN_MASK_DIR,
        transforms=train_transforms)

    val_dataset = FaceDataset(
        images_dir=TRAIN_DATA_DIR,
        images_name=val_images,
        target_dir=TRAIN_MASK_DIR,
        transforms=val_transforms)

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, pin_memory=True)

    unet = model_list[args.model]()

    if torch.cuda.is_available():
        unet = unet.cuda()
        unet = nn.DataParallel(unet, GPUs)

    criterion = Loss(1, 2)
    val_criterion = Loss(0, 1)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    num_epoch = args.num_epoch
    steps = 0

    tmp_losses = []
    print(f'Fold {args.fold}')
    print(f'Steps per epoch: {len(train_data_loader)}')
    print(f'Total steps: {len(train_data_loader)*num_epoch}')

    save_folder = f'{CHECKPOINT_DIR}{args.model}_fold_{args.fold}'
    os.makedirs(save_folder, exist_ok=True)

    for epoch in range(num_epoch):
        for inputs, targets in tqdm(train_data_loader, total=len(train_data_loader)):
            optimizer.zero_grad()
            inputs = cuda(inputs)

            with torch.no_grad():
                targets = cuda(targets)

            output = unet(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            steps += 1
            tmp_losses.append(loss.item())

            if steps % len(train_data_loader) == 0:
                with torch.no_grad():
                    val_loss = []

                    for i, (inputs_val, targets_val) in enumerate(val_data_loader):

                        inputs_val = cuda(inputs_val)

                        targets_val = cuda(targets_val)

                        output = unet(inputs_val)

                        val_loss.append(val_criterion(
                            output, targets_val).item())
                    val_loss = np.mean(val_loss)

                    print(
                        f'steps: {steps},\t' +
                        f'train loss: {round(np.mean(tmp_losses), 4)},\t' +
                        f'val loss: {round(val_loss, 4)}'
                    )

                tmp_losses = []
        save_model(unet, epoch, steps, round(val_loss, 4), save_folder)

        print(f'End epoch {epoch+1} of {num_epoch}')


if __name__ == '__main__':
    main()
