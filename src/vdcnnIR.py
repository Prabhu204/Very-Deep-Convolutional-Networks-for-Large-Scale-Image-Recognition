# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_features, output_features, kernel, padding, stride, conv1_1=False):
        super(ConvBlock).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.conv1_1 = conv1_1
        if conv1_1:
            self.conv = nn.Conv2d(input_features,output_features, kernel_size=1, padding=padding, stride=stride)
        else:
            self.conv = nn.Conv2d(input_features,output_features, kernel_size= kernel, padding=padding, stride= stride)
        self.bNorm= nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bNorm(output)
        output = self.relu(output)
        return output


class Vgg(nn.Module):
    def __init__(self, num_channels, num_classes, max_pool, init_weights, depth, conv1_1= False):
        global num_conv_blocks
        super(Vgg,self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.max_pool = max_pool
        self.init_weights = init_weights
        self.depth = depth
        layers = []
        fc_layers = []
        base_features = 64

        if depth==11:
            num_conv_blocks = [0, 0, 1, 1, 2]
        elif depth== 13:
            num_conv_blocks = [1, 1, 1, 1, 2]
        elif depth == 16:
            num_conv_blocks = [1, 1, 2, 2, 3]
        elif depth==19:
            num_conv_blocks = [1, 1, 3, 3, 4]

        layers.append(ConvBlock(input_features=num_channels, output_features=base_features, kernel=3, padding=1, stride=1))
        for _ in range(num_conv_blocks[0]):
            layers.append(ConvBlock(input_features=base_features, output_features=base_features, kernel=3, padding=1,stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(ConvBlock(input_features=base_features, output_features=2*base_features, kernel=3, padding=1, stride=1))
        for _ in range(num_conv_blocks[1]):
            layers.append(ConvBlock(input_features=2*base_features, output_features=2*base_features, kernel=3, padding=1,stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(ConvBlock(input_features=2*base_features, output_features=4*base_features, kernel=3, padding=1, stride=1))
        for _ in range(num_conv_blocks[2]):
            if conv1_1:
                layers.append(ConvBlock(input_features=4 * base_features, output_features=4 * base_features, kernel=3, padding=1, stride=1, conv1_1=True))
            else:
                layers.append(ConvBlock(input_features=4*base_features, output_features=4*base_features, kernel=3, padding=1,stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(ConvBlock(input_features=4*base_features, output_features=8*base_features, kernel=3, padding=1, stride=1))
        for _ in range(num_conv_blocks[3]):
            layers.append(ConvBlock(input_features=8*base_features, output_features=8*base_features, kernel=3, padding=1,stride=1))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        for _ in range(num_conv_blocks[4]):
            layers.append(ConvBlock(input_features=8*base_features, output_features=8*base_features, kernel=3, padding=1,stride=1))
        layers.append(nn.AdaptiveAvgPool2d(8))
        fc_layers.extend([nn.Linear(in_features=8*8*base_features, out_features= base_features*base_features),nn.ReLU()])
        fc_layers.append([nn.Linear(in_features=base_features*base_features, out_features= base_features*base_features),nn.ReLU()])
        fc_layers.append([nn.Linear(in_features=base_features*base_features, out_features= self.num_classes)])
        self.layers = nn.Sequential(*layers)
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, input):
        output = self.layers(input)
        output = output.view(output.size(0), -1)
        output = self.fc_layers(output)
        return output


