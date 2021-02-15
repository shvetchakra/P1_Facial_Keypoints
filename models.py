## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            ReLU(inplace=True),
            
            #BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.15),
            Conv2d(64, 64, kernel_size=5, stride=1, padding=1),
            ReLU(inplace=True),
            
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            #BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(p=0.15),
            
            Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            # adding batch normalization
            #BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            # adding dropout
            Dropout(p=0.15),
        )

        self.linear_layers = nn.Sequential(
                                         Linear(64 * 6 * 6, 1024),
                                         nn.ReLU(),
                                         nn.Dropout(0.1),
                                         nn.Linear(1024, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(512,136)
        )

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        
        # a modified x, having gone through all the layers of your model, should be returned
        #return x
