# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""


import os
import torch
import argparse
from src.preprocess import preprocess
import numpy as np
import pickle
import pandas as pd
from src.vdcnnIR import Vgg
import matplotlib.image as img
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser("""Very Deep Convolutional Networks for Large Scale Image Recognition""")
    parser.add_argument('-t', '--test', type=str, default='test', help="""required image dataset for training a model.
                                                                              It must be in the data directory """)
    parser.add_argument('-b', '--batchsize', type=int, choices=[64,128,256, 512], default=1024, help='select number of samples to load from dataset')
    parser.add_argument('-i', '--imagesize', type=int, default=64, help="it is used to resize the image pixels" )
    parser.add_argument('-m', '--model', type=int, choices=[11,13,16,19], default=64, help="it is used to resize the image pixels" )
    parser.add_argument('-c11', '--conv1_1', action='store_true', default=False,
                        help="""setting it True will replace some of the 3x3 Conv layers with 1x1 Conv layers in the 16 layer network""")
    args = parser.parse_args()
    return args


def test(opt):
    testdata, testGenerator, classes = preprocess(path='./data' + os.sep + opt.test, batchsize=opt.batchsize,
                                                    imagesize=opt.imagesize , shuffle=False)
    images = [i[0].split('/')[-1] for i in testGenerator.sampler.data_source.imgs]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Vgg(num_channels=3, num_classes=200, depth=opt.model, conv1_1=False,initialize_weights=True).to(device)
    print(iter(testGenerator).__next__()[0].size())
    if opt.conv1_1 and opt.model:
        model = torch.load('./models' + os.sep + 'VdcnnIR_C11_' + str(opt.model))
    else:
        model = torch.load('./models'+os.sep+'VdcnnIR_'+str(opt.model))

    model.eval()
    total_testpredictions = []
    for idx_t, data_t in enumerate(testGenerator):
        data_t, _ = data_t[0], data_t[1]
        data_t = data_t.to(device)
        with torch.no_grad():
            prob_t = model(data_t)
            pred_t = np.argmax(prob_t.detach().cpu(), -1)
            total_testpredictions.extend(pred_t.tolist())
            print('Iter: [{}/{}]'.format(idx_t + 1, len(testGenerator)))

    if opt.conv1_1 and opt.model:
        with open('./results/model_{}_C11_Testpred'.format(opt.model), 'wb') as f:
            pickle.dump(total_testpredictions, f)
    else:
        with open('./results/model_{}_Testpred'.format(opt.model), 'wb') as f:
            pickle.dump(total_testpredictions, f)
    return images, total_testpredictions
if __name__ == '__main__':
    opt = get_args()
    images, pre = test(opt=opt)
    num_models = len(os.listdir('models'))
    with open('results/class_to_idx','rb') as f:
            class_to_id = pickle.load(f)
    if opt.conv1_1 and opt.model:
        df_ = pd.DataFrame({str(opt.model)+'_C11': pre})
    else:
        df_ = pd.DataFrame({str(opt.model):pre})
    if os.path.exists('results/test_predictions.csv'):
        df_old = pd.read_csv('results/test_predictions.csv')
        df_old = pd.concat([df_old, df_], axis=1)
        if len(df_old.columns) == (num_models+1):

            # class prediction of a test image is selected based on mode value of prediction from all models for the same test image.
            df_pre= df_old.mode(axis= 'columns', numeric_only= True)
            df_old = pd.concat([df_old, df_pre], axis=1)
            df_old = df_old.rename(columns={'0': 'mode_pred'})
            df_old['mode_pred'] = df_old['mode_pred'].astype(int)
            class_pred  =[[k for j in df_old.mode_pred.to_list() for k, v in class_to_id.items() if j == v] ]
            df_old = pd.concat([df_old, pd.DataFrame({'class_pred':class_pred})],axis=1)
        df_old.to_csv('results/test_predictions.csv',index=False)

        # display test images with predicted class
        sample = df_old[['images', 'class_pred']].head(n=4)
        images, pred = sample.images.to_list(), sample.class_pred.to_list()
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i, image in enumerate(images):
            axes[i].imshow(img.imread('data/test/images/{}'.format(image)))
            axes[i].set_title('pred_class:' + str(pred[i]))
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        plt.savefig('figures/test_predictions.png')

    elif os.path.exists('results/test_predictions.csv') == False:
        df = pd.DataFrame({'images':images})
        df = pd.concat([df,df_], axis=1)
        df.to_csv('results/test_predictions.csv',index=False)
    print(df_.head())
    print(df_old.head())

