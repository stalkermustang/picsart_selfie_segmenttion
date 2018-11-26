import argparse
import os

import cv2
import numpy as np
import pandas as pd
from scipy.special import expit
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm import tqdm

from config import THRESHOLD, TEST_DATA_DIR
from main_train import model_list
from utils import rle_encode


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--folds', type=str, help='fold', default='0,1,2,3,4,5,6,7,8,9')
    arg('--model', type=str, default='senet154',
        choices=list(model_list.keys()) + ['all', 'top2'])
    
    args = parser.parse_args()

    folds_to_use = [int(i) for i in args.folds.split(',')]

    if args.model == 'all':
        models_to_merge = list(model_list.keys())
    elif args.model == 'top2':
        models_to_merge = ['se_resnext101', 'senet154']
    else:
        models_to_merge = [args.model]

    submit_dir = 'submits/'
    os.makedirs(submit_dir, exist_ok=True)
    subm_name = f'{args.model}_{len(folds_to_use)}_folds_th_{THRESHOLD}.csv'

    path_images = list(filter(lambda x: x.endswith('.jpg'), os.listdir(TEST_DATA_DIR)))
    ids_to_process = [i.split('.')[0] for i in path_images]

    base_folder = 'logits/'

    dirs_to_see = [[f'{base_folder}{model}/fold_{fold}/' 
                        for fold in folds_to_use] 
                                    for model in models_to_merge][0]
    
    predictions = []

    for cur_id in tqdm(ids_to_process):
        tmp_preds = np.zeros((len(dirs_to_see)*2, 256, 256))

        for ind, cur_dir in enumerate(dirs_to_see):
            orig_mask = np.load(f'{cur_dir}id_{cur_id}_normal.npy')
            flip_mask = np.load(f'{cur_dir}id_{cur_id}_tta.npy')

            tmp_preds[ind*2, :, :] = orig_mask
            tmp_preds[ind*2+1, :, :] = flip_mask

        final_pred = expit(np.mean(tmp_preds, axis=0))

        final_img = cv2.resize(final_pred, (240, 320))
        post_img = remove_small_holes(remove_small_objects(final_img > THRESHOLD))
        predictions.append(rle_encode(post_img))

    
    df = pd.DataFrame.from_dict({'image': [i.split('.')[0] for i in path_images], 'rle_mask': predictions})
    df.to_csv(submit_dir+subm_name, index=False)



if __name__ == '__main__':
    main()
