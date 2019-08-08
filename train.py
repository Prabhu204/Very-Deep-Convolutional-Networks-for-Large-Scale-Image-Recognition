# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

import os
import argparse
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import numpy as np
from src.vdcnnIR import Vgg
from src.preprocess import preprocess
import matplotlib.pyplot as plt
import torch.optim as optim
import scikitplot
import shutil
import pickle

def get_args():
    parser = argparse.ArgumentParser("""Very Deep Convolutional Networks for Large Scale Image Recognition""")
    parser.add_argument('-t', '--train', type=str, default='train', help="""required image dataset for training a model.
                                                                           It must be in the data directory """)
    parser.add_argument('-v', '--val', type=str, default='val_', help="""required image dataset for training a model.
                                                                              It must be in the data directory """)
    parser.add_argument('-b', '--batchsize', type=int, choices=[64,128,256], default=50, help='select number of samples to load from dataset')
    parser.add_argument('-e', '--epochs', type=int, choices=[50, 100, 150], default=50)
    parser.add_argument('-d', '--depth', type=int, choices=[11,13,16,19], default=11, help='depth of the deep learning model')
    parser.add_argument('-c11', '--conv1_1', action='store_true', default=False,
                        help="""setting it True will replace some of the 3x3 Conv layers with 1x1 Conv layers in the 16 layer network""")
    parser.add_argument('-es', '--early_stopping', type=int, default= 6, help="""early stopping is used to stop training of network, 
                                                                        if does not improve validation loss""")
    parser.add_argument('-i', '--imagesize', type=int, default=224, help="it is used to resize the image pixels" )
    parser.add_argument('-lr', '--lr', type=int, default=0.001, help="learning rate for an Adam optimizer")
    args = parser.parse_args()
    return args

def train(opt):
    traindata, trainGenerator, classes = preprocess(path='./data'+os.sep+opt.train, batchsize=opt.batchsize,
                                                    imagesize=opt.imagesize, shuffle=True)
    valdata, validationGenerator, classes = preprocess(path='./data'+os.sep+opt.val, batchsize=opt.batchsize,
                                                      imagesize=opt.imagesize, shuffle=True)
    # print(iter(trainGenerator).__next__())

    num_channels = iter(trainGenerator).__next__()[0].size()[1]
    if opt.conv1_1 and opt.depth == 16:
        path_t = 'results/VdcnnIR_train_C11_{}.txt'.format(opt.depth)
        path_v = 'results/VdcnnIR_val_C11_{}.txt'.format(opt.depth)
    else:
        path_t = 'results/VdcnnIR_train_{}.txt'.format(opt.depth)
        path_v = 'results/VdcnnIR_val_{}.txt'.format(opt.depth)
    if os.path.exists(path_t):
        os.remove(path_t)
        os.mknod(path_t)
    else:
        os.mknod(path_t)
    if os.path.exists(path_v):
        os.remove(path_v)
        os.mknod(path_v)
    else:
        os.mknod(path_v)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.conv1_1 and opt.depth==16:
        model = Vgg(num_channels=num_channels,num_classes=classes,depth=opt.depth, initialize_weights=True,
                conv1_1=opt.conv1_1).to(device)

    else:
        model = Vgg(num_channels=num_channels, num_classes=classes, depth=opt.depth, initialize_weights=True,
                    conv1_1=False).to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    def plot_fig(train_loss, val_loss):
        plt.figure(figsize=(10,8))
        plt.title("Train Vs Val loss")
        plt.plot(train_loss, label= 'Train_loss')
        plt.plot(val_loss, label= 'Val_loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        if opt.conv1_1 and opt.depth==16:
            plt.savefig("figures/trainVal_loss_C11_{}.png".format(opt.depth))
        else:
            plt.savefig("figures/trainVal_loss_{}.png".format(opt.depth))

        # plt.show()
        # plt.close()
        return None


    totalVal_loss = []
    totalTrain_loss = []
    early_stop = False
    count = 0
    best_score = None

    for epoch in range(opt.epochs):
        model.train()
        train_loss = []
        total_predictions = []
        total_labels = []
        for idx, data in enumerate(trainGenerator):
            data_, label = data[0], data[1]
            data_ = data_.to(device)
            label = label.to(device)
            # print(data_.size())
            optimizer.zero_grad()
            prob = model(data_)
            # print(prob)
            prob_ = np.argmax(prob.detach().cpu(), -1)
            loss = criterion(prob, label)
            train_loss.append(loss.item()*len(label.cpu()))
            loss.backward()
            optimizer.step()
            total_predictions.extend(prob_)
            total_labels.extend(label.cpu())
            print('Iter: [{}/{}]\t Epoch: [{}/{}]\t Loss: {}\t Acc: {}'.format(idx+1, len(trainGenerator), epoch+1, opt.epochs,
                                                                    loss.item(),
                                                                    metrics.accuracy_score(label.cpu(), prob_)))

        loss_epoch = sum(train_loss)/len(traindata)
        totalTrain_loss.append(loss_epoch)
        with open(path_t, 'a') as f:
            f.write('Epoch{}\tLoss{}\tAccuracy{}\n'.format(epoch+1, loss_epoch,
                                                         metrics.accuracy_score(total_labels,total_predictions)))


        model.eval()
        val_loss = []
        total_Valpredictions = []
        total_ValLabels = []
        for idx_e, data_e in enumerate(validationGenerator):
            data_e,label_e = data_e[0], data_e[1]
            data_e = data_e.to(device)
            label_e= label_e.to(device)
            with torch.no_grad():
                prob_e = model(data_e)
                loss_v = criterion(prob_e, label_e)
                pred_e = np.argmax(prob_e.detach().cpu(),-1)
                val_loss.append(loss_v.item()*len(label_e.cpu()))
                total_ValLabels.extend(label_e.cpu())
                total_Valpredictions.extend(pred_e)
                print('Iter: [{}/{}]\t Epoch: [{}/{}]\t Loss: {}\t Acc: {}'.format(idx_e + 1, len(validationGenerator), epoch + 1,
                                                                        opt.epochs, loss_v.item(),
                                                                        metrics.accuracy_score(label_e.cpu(), pred_e)))
        val_lossEpoch= sum(val_loss)/len(valdata)
        totalVal_loss.append(val_lossEpoch)
        with open(path_v, 'a') as f:
            f.write('Epoch: {}\tLoss: {}\tAccuracy: {}\n'.format(epoch+1, val_lossEpoch,
                                                         metrics.accuracy_score(total_ValLabels,total_Valpredictions)))
        # roc_fig = scikitplot.metrics.plot_roc(total_ValLabels, total_Valpredictions, figsize=(12, 12))
        # plt.savefig('figures/ROC_{}.png'.format(opt.depth))
        # plt.show()
        plot_fig(train_loss=totalTrain_loss, val_loss=totalVal_loss)
        # print(loss_fig)
        if best_score is None:
            best_score = val_lossEpoch
            torch.save(model, 'models/VdcnnIR_{}'.format(opt.depth))
        elif val_lossEpoch > best_score:
            print("Loss:{} doesn't decreased from {}".format(val_loss, best_score))
            count +=1
            if count >= opt.early_stopping:
                early_stop = True
        elif val_lossEpoch < best_score:
            print("Loss:{} decreased from {}. Saving model........".format(val_loss, best_score))
            best_score = val_loss
            torch.save(model, 'models/VdcnnIR_{}'.format(opt.depth))
            count = 0
        if early_stop:
            break
        model.train()
    losses = {'trainLoss':totalTrain_loss, 'valLoss':totalVal_loss}
    if opt.conv1_1 and opt.depth == 16:
        with open('results/losses_C11_{}'.format(opt.depth), 'wb') as f:
            pickle.dump(losses, f)
    else:
        with open('results/losses_{}'.format(opt.depth), 'wb') as f:
            pickle.dump(losses, f)
    return best_score

if __name__ == '__main__':
    opt = get_args()
    loss = train(opt)
    print(loss)


