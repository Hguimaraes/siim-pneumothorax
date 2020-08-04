import cv2
import pydicom
import numpy as np
from torch.utils.data import Dataset
from albumentations import HorizontalFlip
from albumentations import ShiftScaleRotate
from albumentations import Normalize
from albumentations import Resize
from albumentations import Compose
from albumentations import GaussNoise
from albumentations import MultiplicativeNoise
from albumentations.pytorch import ToTensor
from siim_pneumothorax.utils import run_length_decode

"""
Initially based on: https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
"""
class PneumoDataset(Dataset):
    def __init__(self, df, rgb_channel=False, img_size=512, grid_size=32, is_train=True):
        self.df = df
        self.is_train = is_train
        self.img_size = img_size
        self.rgb_channel = rgb_channel
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(grid_size, grid_size))

        # Transforms
        if self.is_train:
            self.transforms = Compose([
                Resize(img_size, img_size),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                GaussNoise(),
                MultiplicativeNoise(),
                ToTensor()
            ])
        else:
            self.transforms = Compose([
                Resize(img_size, img_size),
                ToTensor()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img = pydicom.read_file(item.fn).pixel_array
        img = self.clahe.apply(img)/255.

        if (len(item.EncodedPixels) == 1) and (item.EncodedPixels[0] == "-1"):
            masks_list = np.zeros((self.img_size, self.img_size))

        else:
            masks_list = np.sum(np.array([
                run_length_decode(mask, 1024, 1024) for mask in item.EncodedPixels]), 
                axis=0
            )
            masks_list[masks_list > 0] = 1

        # Insert channel
        img = np.expand_dims(img, axis=-1)

        if self.rgb_channel:
            img = np.concatenate((img, img, img), axis=2)

        transformed = self.transforms(image=img, mask=masks_list)
        img, masks_list = transformed['image'], transformed['mask']

        return img, masks_list