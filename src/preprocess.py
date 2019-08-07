# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""
import os
import torchvision.transforms as tf
from torchvision.datasets import DatasetFolder, ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def preprocess(path, batchsize, imagesize, shuffle = True):
    transform = tf.Compose([tf.Scale((imagesize,imagesize)),
                            tf.RandomHorizontalFlip(),
                            tf.ToTensor(),
                            tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    loaddata= ImageFolder(root=path, transform=transform)
    dataLoader = DataLoader(dataset=loaddata, batch_size= batchsize, shuffle=shuffle)
    classes_ = loaddata.classes
    images, class_ = iter(dataLoader).__next__()
    def plot_img(img):
        img = img/2 +0.5
        img = img.numpy()
        return img.transpose(1,2,0)
    fig, axes  =plt.subplots(1, 4, figsize=(12,3))
    for i, img in enumerate(images):
        axes[i].imshow(plot_img(img))
        axes[i].set_title(classes_[class_[i]])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        # print(class_[i])
        if i == 3:
            break
    if 'train' in path:
        plt.savefig('figures/train_sample.png')
        plt.close()
    else:
        plt.savefig('figures/val_sample.png')
        plt.close()
    return loaddata, dataLoader, len(classes_)