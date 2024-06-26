# -*- coding: utf-8 -*-
"""EMNIST_Digit_Preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1REzMdtU4HBReBInICjB70C78cugMq5We
"""

from einops import rearrange, repeat

import json
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.v2 as transforms_v2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import time

import torch.optim as optim
import torchvision.datasets as dsets
import matplotlib.pylab as plt
torch.manual_seed(2)
import torchsummary
import os

from tqdm import tqdm
import requests
import gzip
import os
import numpy as np

def download_emnist(dir_root,file_dict=None):
    if file_dict is not None:
        mnist_data=list()
        try:
            for i, key in enumerate(file_dict.keys()):
                fname = file_dict[key]
                file_path = os.path.join(dir_root,fname)

                isExist = os.path.exists(fname)


                with gzip.open(fname, "rb") as f_in:
                    if fname.find('idx3') != -1:
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=16).reshape(-1,28, 28)) #if images
                    else:
                        mnist_data.append(np.frombuffer(f_in.read(), np.uint8, offset=8))  #if labels
            #return mnist_data in a list format ==> [[test_images], [test_labels]]
            return mnist_data
        except Exception as e:
            print("Something went wrong:", e)
    else:
        print("file_dict cannot be None")

# dir_root = '/content/'
# file_dict={
#    'train_images':'emnist-letters-train-images-idx3-ubyte.gz',
#    'train_labels':'emnist-letters-train-labels-idx1-ubyte.gz',
#    'test_images':'emnist-letters-test-images-idx3-ubyte.gz',
#    'test_labels':'emnist-letters-test-labels-idx1-ubyte.gz'
#}
#dataset= download_emnist(dir_root,file_dict)

class MNISTCustomDataset(Dataset):

    def __init__(self, images, labels, transform=None, * , label_type):
        super().__init__()
        assert label_type in {'string', 'integer'}, 'label type must be either a string or an integer between 0 to 9'
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_type = label_type

    def __getitem__(self, idx):
        labels_mapping = {
                          0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                          }

        label = self.labels[idx]


        new_label = label

        image = self.images[idx]
        image = self.transform(image)
        #image = rearrange(image,'1 h w -> h w')
        return image, new_label
    def __len__(self):
        return len(self.labels)

