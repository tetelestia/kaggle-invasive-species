"""
File containing all code pertaining to the dataset and augmentations
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable


class InvasiveDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle's
       Invasive Species Monitoring Competition

    Arguments:
        A CSV file path
        Path to image folder
        Requested image transformations
    """

    def __init__(self, label_csv, img_path, img_size, transform=None):

        self.label_csv= label_csv
        self.img_path = img_path
        self.img_size = img_size
        self.transform = transform

        # Load csv into pandas dataframe
        self.data_df = pd.read_csv(self.label_csv)

        # Ensure all images exist before we begin training
        self.check_dataset()

        # Populate X and y vectors
        self.X_train = self.data_df['name']
        self.y_train = self.data_df['invasive']

    def __getitem__(self, index):
        """Load single image, given its index"""
        # Load image from disk
        img = Image.open('{}{}.jpg'.format(self.img_path, self.X_train[index]))
        img = img.resize(self.img_size)

        # Apply transformation if requested
        if self.transform is not None:
            img = self.transform(img)
    
        # Set label from y_train
        label = self.y_train[index].astype(np.float32)

        img = np.transpose(img, (2,0,1))
        img = torch.from_numpy(img).float()

        # Return image and label to data loader
        return img, label

    def __len__(self):
        return len(self.X_train.index)

    def check_dataset(self):
        assert self.data_df['name'].apply(
            lambda x: os.path.isfile('{}{}.jpg'.format(self.img_path, x))).all(), \
            "Some images in CSV file were not found on disk."
