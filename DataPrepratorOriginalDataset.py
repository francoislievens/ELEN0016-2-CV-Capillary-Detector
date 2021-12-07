import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import time


def prepare_data_folders(data_path):
    os.mkdir(data_path)
    if not os.path.exists('{}/annotations'.format(data_path)):
        os.mkdir('{}/annotations'.format(data_path))
    if not os.path.exists('{}/images'.format(data_path)):
        os.mkdir('{}/images'.format(data_path))


def clean_data(path, dir):
    df = pd.read_csv(path+"/"+dir+"/"+dir+".csv", sep=";")
    # Terms
    # nb_slice = df.apply(pd.value_counts)
    # print(nb_slice)
    # nb_slice = df.groupby('Terms').count()
    # print(nb_slice)
    # print(df.groupby('Track').size()[0])
    # print(df.groupby('Track').size())
    return len(df.groupby('Track').size())-1


def data_preparator_original_dataset(data_path):
    nb_tracks = []
    for dir in os.listdir(data_path+'/annotations'):
        nb_tracks += clean_data(data_path+'/annotations', dir)


data_preparator_original_dataset('Original_Dataset')
