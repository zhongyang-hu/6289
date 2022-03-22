# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import math
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import torch
from torch import nn, optim
import timm

import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split




import warnings

path = "../midProject1/dataset/dataset/"


from model2 import efficientnet_b1, get_data_loaders, get_classes, train_model







train_loader, test_loader ,train_len,test_len= get_data_loaders(path,40)



dataloads = {
    "train":train_loader,
    "val": test_loader}
dataset_sizes = {
    "train":train_len,
    "val": test_len}








model = efficientnet_b1(n_classes=151)

device = torch.device("cuda")
print(device)
model = model.to(device)
device

criterion = nn.CrossEntropyLoss()
print(criterion)
criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-5)

training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}

from tqdm import tqdm
import time
import copy

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)








model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,dataloads,device,dataset_sizes,training_history ,validation_history,'compare.pt',
                       num_epochs=1)



