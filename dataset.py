import cv2
import os
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, images_dir, images_name, target_dir=None,
                 transforms=None):
        
        self.images_dir = images_dir
        self.target_dir = target_dir
        self.images_name = images_name
        self.transforms = transforms
                           
        print('{} images'.format(len(self.images_name)))

    def __len__(self):
        return len(self.images_name)
               
    def __getitem__(self, idx):
        img_filename = os.path.join(self.images_dir, self.images_name[idx])
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.target_dir:
            mask_filename = os.path.join(
                self.target_dir, self.images_name[idx].replace('.jpg', '.png'))
            mask = cv2.imread(mask_filename, 0)
        else:
            mask = []

        if self.transforms:
            if mask!=[]:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transforms(image=img)
                img = augmented['image']

        return img, mask
