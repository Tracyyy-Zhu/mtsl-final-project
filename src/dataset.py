import os
from os.path import isfile, join, abspath, exists, isdir, expanduser
from os import listdir, makedirs, getcwd, remove
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

import pandas as pd
import numpy as np

#########################
##   Dataset Classes   ##
#########################

class SeedlingDataset(Dataset):
    def __init__(self, df, path, small_sample=False, transform=None, use_train_folder=True):
        self.df = df
        self.path = path
        self.transform = transform
        if small_sample:
            self.df = self.df.sample(50)
        if use_train_folder:
            self.path += "train/"
        else:
            self.path += "test/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        img = Image.open(self.path + img_path).convert("RGB")
        label = self.df.iloc[idx, 2]

        if self.transform:
            img = self.transform(img)

        return img, label

#########################
##  Data Augmentation  ##
#########################

class IdentityTransform(object):
    
    def __call__(self, data):
        return data

def get_train_trans(image_size=224, data_aug = IdentityTransform()):
    """
    Transform function for processing images in the training set.
    """

    return transforms.Compose([
        # TODO Fit the data aug function in
        data_aug,
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
def get_val_trans(image_size=224):
    """
    Transform function for processing images in the validation/test set.
    """
    return transforms.Compose([
        # TODO Fit the data aug function in
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    



#################
##  Functions  ##
#################

def get_classes_indices_mapping(data_dir):
    """
    Input the data folder directory.
    
    Return list of classes, mapping from class name to index, and backwards.
    """
    classes = sorted([d for d in os.listdir(data_dir + "train/")])
    class_to_idx = {cls:i for i, cls in enumerate(classes)}
    idx_to_class = {i:cls for i, cls in enumerate(classes)}
    
    return classes, class_to_idx, idx_to_class


def get_data_df(data_dir):
    """
    Input the data folder directory.
    
    Return a pandas DataFrame of the image paths, their corresponding labels, and label indices.
    """
    classes, class_to_idx, idx_to_class = get_classes_indices_mapping(data_dir)
    
    data_l = []
    for idx, label in enumerate(classes):
        path_to_image = data_dir + "train/" + label
        for file in listdir(path_to_image):
            data_l.append([f"{label}/{file}", label, idx])

    data_df = pd.DataFrame(data_l, columns=["filename", "label", "index"])
    
    return data_df

def get_train_val_loader(data_dir, val_ratio=0.2, train_trans=None, val_trans=None,
                        batch_size=32, small_sample=False):
    """
    Generate the train and validation dataloaders.
    """
    
    train_df, val_df = train_test_split(get_data_df(data_dir), test_size=val_ratio)
    
    train_set = SeedlingDataset(train_df, data_dir, small_sample=small_sample,
                                transform = train_trans)
    val_set = SeedlingDataset(val_df, data_dir, small_sample=small_sample,
                              transform = val_trans)
    
    return ( DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=True) )
    
def test_image_loader(im_dir, trans):
    """
    Image loader for each test set image
    """
    im = Image.open(im_dir).convert("RGB")
    im = trans(im)
    im = im.unsqueeze(0)
    return im
