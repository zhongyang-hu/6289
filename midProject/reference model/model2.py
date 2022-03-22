

import torch
from torch import nn, optim
import timm

import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split

torchvision.__version__, torch.__version__ # ('0.11.1+cu111', '1.10.0+cu111')

import warnings
warnings.filterwarnings("ignore")

import math



from tqdm import tqdm
import time
import copy


def get_data_loaders(img_dir, batch_size):
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(256),
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
    return train_loaded, test_loaded,train_data_num,test_data_num

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

torch.backends.cudnn.benchmark = True

def conv_block(in_channels, out_channels, kernel_size=3,
               stride=1, padding=0, groups=1,
               bias=False, bn=True, act = True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=padding, groups=groups, bias=bias),
        nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
        nn.SiLU() if act else nn.Identity()
    ]
    return nn.Sequential(*layers)

class SEBlock(nn.Module):
    def __init__(self, c, r=24):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(c, c // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        s = self.squeeze(x)
        e = self.excitation(s)
        return x * e


class MBConv(nn.Module):
    def __init__(self, n_in, n_out, expansion, kernel_size=3, stride=1, r=24, dropout=0.1):
        super(MBConv, self).__init__()
        self.skip_connection = (n_in == n_out) and (stride == 1)
        padding = (kernel_size - 1) // 2
        expanded = expansion * n_in

        self.expand_pw = nn.Identity() if expansion == 1 else conv_block(n_in, expanded, kernel_size=1)
        self.depthwise = conv_block(expanded, expanded, kernel_size=kernel_size,
                                    stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r)
        self.reduce_pw = conv_block(expanded, n_out, kernel_size=1, act=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)
        if self.skip_connection:
            x = self.dropout(x)
            x = x + residual
        return x

base_widths = [(32, 16), (16, 24), (24, 40),
                   (40, 80), (80, 112), (112, 192),
                   (192, 320), (320, 1280)]
base_depths = [1, 2, 2, 3, 3, 4, 1]
kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
strides = [1, 2, 2, 2, 1, 2, 1]
ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]


def mbconv1(n_in, n_out, kernel_size=3, stride=1, r=24, dropout=0.1):
    return MBConv(n_in, n_out, 1, kernel_size=kernel_size, stride=stride, r=r, dropout=dropout)


def mbconv6(n_in, n_out, kernel_size=3, stride=1, r=24, dropout=0.1):
    return MBConv(n_in, n_out, 6, kernel_size=kernel_size, stride=stride, r=r, dropout=dropout)

def create_stage(n_in, n_out, num_layers, layer=mbconv6,
                 kernel_size=3, stride=1, r=24, ps=0):
    layers = [layer(n_in, n_out, kernel_size=kernel_size,
                       stride=stride, r=r, dropout=ps)]
    layers += [layer(n_out, n_out, kernel_size=kernel_size,
                        r=r, dropout=ps) for _ in range(num_layers-1)]
    return nn.Sequential(*layers)
def scale_width(w, w_factor):
    w *= w_factor
    new_w = (int(w+4) // 8) * 8
    new_w = max(8, new_w)
    if new_w < 0.9*w:
        new_w += 8
    return int(new_w)

def efficientnet_gen(w_factor=1, d_factor=1):
    scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor))
                     for w in base_widths]
    scaled_depths = [math.ceil(d_factor*d) for d in base_depths]
    return scaled_widths, scaled_depths


class EfficientNet(nn.Module):
    def __init__(self, w_factor=1, d_factor=1, n_classes=1000):
        super(EfficientNet, self).__init__()
        scaled_widths, scaled_depths = efficientnet_gen(w_factor=w_factor, d_factor=d_factor)

        self.conv1 = conv_block(3, scaled_widths[0][0], stride=2, padding=1)

        stages = [
            create_stage(*scaled_widths[i], scaled_depths[i], layer=mbconv1 if i == 0 else mbconv6,
                         kernel_size=kernel_sizes[i], stride=strides[i], r=4 if i == 0 else 24, ps=ps[i]) for i in
            range(7)
        ]
        self.stages = nn.Sequential(*stages)
        self.pre = conv_block(*scaled_widths[-1], kernel_size=1)
        self.pool_flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(
            nn.Linear(scaled_widths[-1][1], n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.pre(x)
        x = self.pool_flatten(x)
        x = self.head(x)
        return x

def efficientnet_b0(n_classes=1000):
    return EfficientNet(n_classes=n_classes)
def efficientnet_b1(n_classes=1000):
    return EfficientNet(1, 1.1, n_classes=n_classes)
def efficientnet_b2(n_classes=1000):
    return EfficientNet(1.1, 1.2, n_classes=n_classes)
def efficientnet_b3(n_classes=1000):
    return EfficientNet(1.2, 1.4, n_classes=n_classes)
def efficientnet_b4(n_classes=1000):
    return EfficientNet(1.4, 1.8, n_classes=n_classes)
def efficientnet_b5(n_classes=1000):
    return EfficientNet(1.6, 2.2, n_classes=n_classes)
def efficientnet_b6(n_classes=1000):
    return EfficientNet(1.8, 2.6, n_classes=n_classes)
def efficientnet_b7(n_classes=1000):
    return EfficientNet(2, 3.1, n_classes=n_classes)


def train_model(model, criterion, optimizer, scheduler, dataloaders,device, dataset_sizes, training_history ,validation_history,savenet,num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                training_history['accuracy'].append(epoch_acc)
                training_history['loss'].append(epoch_loss)
            elif phase == 'val':
                validation_history['accuracy'].append(epoch_acc)
                validation_history['loss'].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print([float(i) for i in training_history['accuracy']])
        print([float(i) for i in validation_history['accuracy']])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model