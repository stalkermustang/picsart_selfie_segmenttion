import numpy as np
import torch

from sklearn.model_selection import ShuffleSplit

from config import N_FOLDS, VAL_SPLIT_SIZE

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(320, 240)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T


def get_split(imgs_list, split_num):
    kf = ShuffleSplit(n_splits=N_FOLDS, test_size=VAL_SPLIT_SIZE, random_state=42)

    splits = kf.split(imgs_list)

    idx_train = None
    idx_val = None
    for ind, fld in enumerate(splits):
        if ind == split_num:
            idx_train = fld[0]
            idx_val = fld[1]

    train_images, val_images = imgs_list[idx_train], imgs_list[idx_val]

    return train_images, val_images

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def save_model(model, ep, step, loss, folder):
     torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, f'{folder}/{ep}_{str(loss).split(".")[1]}.pt')
