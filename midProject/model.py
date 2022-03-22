import torch

import pandas as pd

from torch import nn, optim
import copy

import time
import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
import math
def load(img_dir, batch_size):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.488), (0.2172))
        ])
    img_data = datasets.ImageFolder(img_dir,transform=t)
    train_data_num = int(len(img_data)*0.8)

    test_data_num = int(len(img_data) - train_data_num)
    train_data, test_data = random_split(img_data, [train_data_num, test_data_num])
    train_loaded = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loaded =DataLoader(test_data, batch_size=1)
    return train_loaded, test_loaded





class CNN1(nn.Module):
    def __init__(self,CLmake,FLmake):
        super().__init__()
        self.Clayers=CLmake

        self.Flayers=FLmake


    def forward(self,x):
        x= self.Clayers(x)
        x= torch.flatten(x,1)
        x= self.Flayers(x)

        return x




def CLmake(x,size):
    layers=[]
    for  i in x:
        if i[0]==1:
            print(i,'add conv2d')
            conv2d=nn.Conv2d(*i[1])
            layers.append(conv2d)

        if i[0]==2:
            print(i,'add relu6')

            layers.append(nn.ReLU6(True))

        if i[0]==3:
            print(i,'add maxpool')
            layers.append(nn.MaxPool2d(*i[1]))
        if i[0]==4:
            print(i,'add relu')

            layers.append(nn.ReLU(True))
        if i[0]==5:
            print(i,'add conv2d')
            conv2d=nn.Conv2d(*i[1])
            layers.append(conv2d)



    print('size of final product ',size)
    layers.append(nn.AdaptiveAvgPool2d(size))
    return nn.Sequential(*layers)


def FLmake(x):
    layers=[]
    for  i in x:
        if i[0]==1:
            layers.append(nn.Linear(*i[1]))
            print(i,'add full connected')
        if i[0]==2:
            print(i, 'add relu')
            layers.append(nn.ReLU(True))

        if i[0]==3:
            print(i, 'add dropout')
            layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)
#[64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
CLconfig=[[1,[3,64,3]],[2],[3,[2,2]],
          [1,[64,128,3]],[2],[3,[2,2]],
          [1,[128,256,3]],[2],[3,[2,2]],
          [1,[256,512,3]],[2],[3,[2,2]],
          [1,[512,512,3]],[2],[3,[2,2]]]
FLconfig=[[1,[512*49,4096]],[2],[3],
          [1,[4096,2048]],[2],[3],
          [1,[2048,152]]]


def train_model(net, criterion, optimizer,  train_loaded, test_loaded, num_epochs,save_name,savenet,device='cuda',scheduler=None):
    accTrack=pd.DataFrame()
    recordTime=[]
    accTrain=[]
    accTest=[]
    maxAcc=0
    start=time.time()
    net.to(device)
    criterion.to(device)
    for epoch in range(num_epochs):  # loop over the dataset multiple times


        net.train()
        for i, data in enumerate(train_loaded, 0):

            inputs, labels = data

            inputs, labels = inputs.cuda(), labels.cuda()
            net.train()

            optimizer.zero_grad()


            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler != None:
            scheduler.step()
        net.eval()
        accTrain.append(acc(net, train_loaded))
        temp=acc(net, test_loaded)
        accTest.append(temp)
        if temp > maxAcc:
            maxAcc=temp
            bestPara = copy.deepcopy(net.state_dict())

        print(time.time() - start)
        recordTime.append(time.time() - start)

    accTrack['accTrain']=accTrain
    accTrack['accTest']=accTest
    accTrack['time']=recordTime
    net.load_state_dict(bestPara)
    model_scripted = torch.jit.script(net)
    model_scripted.save(savenet)
    accTrack.to_csv(save_name)

    return net



def acc(n,testdata):

    correct = 0
    total = 0
    n.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testdata:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            # calculate outputs by running images through the network
            outputs = n(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total} %')
    return correct/total