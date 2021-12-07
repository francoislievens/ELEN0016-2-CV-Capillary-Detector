import os
from random import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import shuffle_lst, heat_map_gen
import cv2
import pandas as pd
import sys
import albumentations as A
import matplotlib.pyplot as plt
from utils import count_peaks_2d

class DataBuilder(Dataset):

    def __init__(self,
                 dataset_path='UNet_Dataset',
                 train_prop=0.8,
                 train=True,
                 data_augmentation=True):

        super(DataBuilder, self).__init__()
        self.set_seed = 1       # Seed to split the train and test set
        self.dataset_path = dataset_path
        self.data_augmentation = data_augmentation

        # Get images list
        img_lst = os.listdir('{}/Images'.format(dataset_path))

        # Get annotation list
        ann_lst = os.listdir('{}/Annotations'.format(dataset_path))

        # Get the proportion with/without to increase the number of with cell inputs
        report = int(len(img_lst)/len(ann_lst))

        # Get list of elements with and without cells
        with_cell = []
        no_cell = []
        for inpt in img_lst:
            inpt = inpt.replace('.jpg', '')
            if '{}.txt'.format(inpt) in ann_lst:
                with_cell.append(inpt)

            else:
                no_cell.append(inpt)

        # Shuffle both list
        with_cell = shuffle_lst(with_cell, seed=self.set_seed)
        no_cell = shuffle_lst(no_cell, seed=self.set_seed)

        # Split in train and test sets
        with_cell_split_idx = int(train_prop * len(with_cell))
        no_cell_split_idx = int(train_prop * len(no_cell))
        with_cell_train = with_cell[0:with_cell_split_idx]
        with_cell_test = with_cell[with_cell_split_idx:]
        no_cell_train = no_cell[0:no_cell_split_idx]
        no_cell_test = no_cell[no_cell_split_idx:]

        # Ballance the dataset
        with_cell_train = with_cell_train * report
        with_cell_test = with_cell_test * report

        # Concatenate both arrays
        data_train = with_cell_train + no_cell_train
        data_test = with_cell_test + no_cell_test

        # Shuffle again:
        data_train = shuffle_lst(data_train, seed=self.set_seed + 1)
        data_test = shuffle_lst(data_test, seed=self.set_seed + 2)

        self.dataset = data_train
        if not train:
            self.dataset = data_test

        # Memorize which element have a label
        self.ann_lst = ann_lst

        # Data augmentation object
        self.transform = A.Compose(
            [
                A.Rotate(limit=45, p=0.5),        # Rotation max 45 degrees in 50% of the cases
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index: int):


        # Load the image, resize it and convert to grayscale using openCV
        img = cv2.imread('{}/Images/{}.jpg'.format(self.dataset_path, self.dataset[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load annotation file if exist
        ann = []
        if '{}.txt'.format(self.dataset[index]) in self.ann_lst:
            tmp = pd.read_csv('{}/Annotations/{}.txt'.format(self.dataset_path, self.dataset[index]), sep=' ', header=None)
            for i in range(0, tmp.shape[0]):
                # Convert coordinates to pixels relative
                ann.append([tmp.iloc[i][1] * img.shape[1], tmp.iloc[i][2] * img.shape[0]])

        # Create the density map
        label = heat_map_gen(ann, img.shape)

        # Apply data augmentation
        if self.data_augmentation:
            augmented = self.transform(image=img, mask=label)

            return augmented['image'], augmented['mask']

        return img, label

def nb_cell_histogram():

    # Get the total number of droplets in the dataset:
    img_lst = os.listdir('UNet_Dataset/Images')
    tot_frame = len(img_lst)

    hist = np.zeros(7)
    # Read each annotation files to construct the histogram
    ann_list = os.listdir('UNet_Dataset/Annotations')
    for itm in ann_list:
        df = pd.read_csv('UNet_Dataset/Annotations/{}'.format(itm), sep=' ', header=None)
        nb_cell = df.shape[0]
        if nb_cell == 0:
            print(df)
        hist[nb_cell] += 1

    # Compute the number of droplets with zero cell
    no_cell = tot_frame - np.sum(hist)
    print(hist)
    hist[0] = no_cell

    print(hist)
    print('Total nb frames : {}'.format(tot_frame))

    for i in range(0, len(hist)):
        print('{} cell droplets: {}%, - with {} frames'.format(i, (hist[i]/tot_frame)*100, hist[i]))

if __name__ == '__main__':

    nb_cell_histogram()

    dataset = DataBuilder(data_augmentation=False)

    # Get an index lst to test
    idx_lst = np.arange(0, 100)
    np.random.shuffle(idx_lst)

    for idx in idx_lst:

        print('test dataset at idx {}'.format(idx))
        img, target = dataset.__getitem__(idx)

        # Makes the sum on columns
        count_peaks_2d(target)


        print(img.shape)
        print('sum: {}'.format(np.sum(target)))
        # Show in two windows
        cv2.imshow('img', img)
        cv2.imshow('target', target)
        cv2.waitKey()



