from model import CNN1,FLmake, CLmake, train_model, acc, load

import torch

import pandas as pd

from torch import nn, optim
import copy

import time
import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split

img_dir='../midProject1/dataset/dataset/'
train_loaded,test_loaded=load(img_dir,30)
CLconfig=[[5,[3,64,5,1,2]],[4],[3,[2,2]],
          [5,[64,128,3,1,1]],[4],[3,[2,2]],
          [5,[128,256,3,1,1]],[4],[3,[2,2]],
          [5,[256,256,3,1,1]],[4],[3,[2,2]],
          [5,[256,256,3,1,1]],[4],[3,[2,2]]]

FLconfig=[[1,[256*49,2048]],[2],[3],
          [1,[2048,2048]],[2],[3],
          [1,[2048,151]]]

net1=CNN1(CLmake(CLconfig,(7,7)),FLmake(FLconfig))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net1.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5)

accCNN=[]
device='cuda'
net1=net1.to(device)
criterion=criterion.to(device)
train_model(net1,criterion,optimizer,train_loaded,test_loaded,60, 'trainNetC.csv','trainNetC.pt')