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


classes  = os.listdir('data/train')

transform = tf.Compose([tf.Scale((224,224)),
                        tf.RandomHorizontalFlip(),
                        tf.ToTensor(),
                        tf.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


traindata = ImageFolder(root='./data/train', transform=transform)
trainLoader = DataLoader(dataset=traindata, batch_size= 128, shuffle=True)

images, class_ = iter(trainLoader).__next__()

def plot_img(img):
    img = img/2 +0.5
    img = img.numpy()
    return img.transpose(1,2,0)

fig, axes  =plt.subplots(1, 4, figsize=(12,3))

for i, img in enumerate(images):
    axes[i].imshow(plot_img(img))
    axes[i].set_title(classes[class_[i]])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    if i== 3:
        break
plt.savefig('xxx.png')