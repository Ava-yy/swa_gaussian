"""
    separate data loader for imagenet
"""

import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_folder import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from PIL import Image
from PIL import ImageFile
from glob import glob
from tqdm import tqdm
import json

from sklearn.model_selection import train_test_split



IMG_SIZE = (224, 224) 
BATCH_SIZE = 1 
BATCH_SIZE_TRAIN = 8
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
BASE_PATH = '../../food-101/'
TEST_SPLIT = 0.1
epochs = 15
CHOOSED_CLASSES = ['french_toast', 'greek_salad', 'caprese_salad', 'chocolate_cake', 'pizza', 'cup_cakes', 'carrot_cake','cheesecake','pancakes', 'strawberry_shortcake']


class myDataset(torch.utils.data.Dataset):

    def __init__(self, image_df, mode='train'):
        self.dataset = image_df
        self.CHOOSED_CLASSES = CHOOSED_CLASSES

        train_transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])
        if mode == 'train':
            self.transform = train_transforms
        else:
            self.transform = val_transforms

    def __getitem__(self, index):

        c_row = self.dataset.iloc[index]

        image_id = c_row['image_id']

        image_path, target = c_row['path'], self.CHOOSED_CLASSES.index(c_row['category'])  #image and target
        image = Image.open(os.path.join(BASE_PATH,image_path))

        image = self.transform(image)
        
        return image, int(target),c_row['category'], image_path,image_id

    def __len__(self):
        return self.dataset.shape[0]


# train_df, test_df = train_test_split(all_img_df_selected,
#                                      test_size=TEST_SPLIT,
#                                      random_state=42,
#                                      stratify=all_img_df_selected['category'])

# print('train', train_df.shape[0], 'test', test_df.shape[0])

# train_df.to_csv('finetune_food101_train.csv')
# test_df.to_csv('finetune_food101_test.csv')

# # train_df = pd.read_csv('finetune_food101_train.csv')
# # test_df = pd.read_csv('finetune_food101_test.csv')


def data_loaders(train_df_path, test_df_path, train_batch_size, test_batch_size):

    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    train_dataset = myDataset(train_df, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                    batch_size=train_batch_size, pin_memory=False)

    test_dataset = myDataset(test_df, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                    batch_size=test_batch_size, pin_memory=False)

    return {"train": train_loader,"test": test_loader}

