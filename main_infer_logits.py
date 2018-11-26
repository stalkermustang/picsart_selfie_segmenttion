import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from augmentations import test_flip_transforms, test_norm_transforms
from config import N_FOLDS
from dataset import FaceDataset
from main_train import model_list
from tqdm import tqdm
from utils import cuda

BS = 12

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--device_ids', type=str, default='0')
    arg('--folds', type=str, help='fold', default='0,1,2,3,4,5,6,7,8,9')
    arg('--model', type=str, default='se_resnext101',
        choices=list(model_list.keys()))
    arg('--batch_size', type=int, default=BS)

    args = parser.parse_args()
    GPUs = [int(i) for i in args.device_ids.split(',')]
    folds_to_use = [int(i) for i in args.folds.split(',')]

    path_images = list(filter(lambda x: x.endswith('.jpg'), os.listdir('../data/test/')))

    unet_base_model = model_list[args.model]

    test_data_loader_normal = DataLoader(
        FaceDataset('../data/test', path_images, transforms=test_norm_transforms), 
        batch_size=args.batch_size, num_workers=10, shuffle=False)

    test_data_loader_flip = DataLoader(
        FaceDataset('../data/test', path_images, transforms=test_flip_transforms), 
        batch_size=args.batch_size, num_workers=10, shuffle=False)
    
    savedir_base = f'logits/{args.model}/'
    os.makedirs(savedir_base, exist_ok=True)
    
    for cur_fold_num in folds_to_use:
        search_dir = f'checkpoints/{args.model}_fold_{cur_fold_num}/'
        files_in_dir = os.listdir(search_dir)
        scores = [float('0.'+i.split('_')[1].split('.')[0]) for i in files_in_dir]
        chckp_to_use = files_in_dir[np.argmin(scores)]
        chkp_pth = f'{search_dir}{chckp_to_use}'

        print('use checkpoint ', chkp_pth)
        savedir = f'{savedir_base}fold_{cur_fold_num}/'
        os.makedirs(savedir, exist_ok=True)

        unet = unet_base_model(pretrained=False)

        if torch.cuda.is_available():
            unet = unet.cuda()
            unet = nn.DataParallel(unet, GPUs)

        unet.load_state_dict(torch.load(chkp_pth)['model'])
        unet.eval()

        img_cntr = 0
        with torch.no_grad():
            for batch_n, batch_f in tqdm(
                                zip(test_data_loader_normal, 
                                    test_data_loader_flip), 
                                    total=len(test_data_loader_normal)
                                    ):
                inp_n = cuda(batch_n[0])
                inp_f = cuda(batch_f[0])

                output_n = unet.forward(inp_n)
                output_f = unet.forward(inp_f)

                for img_batch_index in range(output_n.shape[0]):
                    img_n = output_n[img_batch_index].cpu().numpy()[0]
                    img_f = output_f[img_batch_index].cpu().numpy()[0]
                    img_f = np.fliplr(img_f)

                    img_id = path_images[img_cntr].split('.')[0]

                    np.save(f'{savedir}/id_{img_id}_normal',img_n)
                    np.save(f'{savedir}/id_{img_id}_tta',img_n)
                    
                    img_cntr += 1

if __name__ == '__main__':
    main()
