from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable
from collections import defaultdict

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--epoch', required=True, type=int, help='epoch number')
args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Test only option
print('\n[Test Phase] : Model setup')
assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
_, file_name = getNetwork(args)
if use_cuda:
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+"-epoch-"+str(args.epoch)+'.t7')
else:
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+"-epoch-"+str(args.epoch)+'.t7', map_location='cpu')
net = checkpoint['net']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

net.eval()
test_loss = 0
correct = 0
total = 0

model = nn.Sequential(*list(net.features.children()))
layer_data = defaultdict(list)
for batch_idx, (inputs, targets) in enumerate(trainloader):
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)

    outputs = model(inputs).data.numpy()

    for idx, output in enumerate(outputs):
        layer_data[targets.data.numpy()[idx]].append(output)

for key in layer_data:
    layer_data[key] = np.mean(np.array(layer_data[key])[:, :, 0, 0], axis=0)

plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(10, 1, i+1)
    mn = layer_data[i].min()
    mx = layer_data[i].max()
    plt.ylim([mn, mx])
    plt.xticks([])
    plt.yticks([])
    plt.bar(range(layer_data[i].shape[0]), layer_data[i])
    plt.ylabel(cf.classes[i])
    if i == 0:
        plt.title("Epoch {}".format(args.epoch))
plt.savefig("penultimate layer - epoch - {}.png".format(args.epoch))
