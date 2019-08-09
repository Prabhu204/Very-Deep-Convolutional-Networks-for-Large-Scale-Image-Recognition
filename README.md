


# Very-Deep-Convolutional-Networks-for-Large-Scale-Image-Recognition

This project work is a implementation of [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) in Pytorch. 
However, the dataset used for this project is from [Tiny ImageNet Visual Recognition Challenge](https://tiny-imagenet.herokuapp.com/). Model performance can be 
improved by using smaller convolutional filter
(i.e feature extraction maps) with 3x3 size while increasing network layer depth.  

### Model Architecture
The models were implemented as per original paper description, which can be seen in the following figure.

![Model Architecture](figures/ConNet_config.png)

### Dataset

The dataset is devided in to 3 sets:

- Trainset is composed with 200 clasess and each class has 200 samples with a 64x64 pixels
- Validationset has composed with a 50 samples per each class. 
- Testset is a total size 10000 samples.

<b>Sample images from trainset:</b>

![trainsample](figures/train_sample.png)

<b>Sample images from validationset:</b>

![valsample](figures/val_sample.png)


### Computing resources 
The models were trained on GPU, which is having a configuration of Nvidia-GTX 1070 8gb. Each model has took approx. 14h to 20h for training.  

### Results

<b>Model performance plots:</b>

<!--![11 layer model](figures/trainVal_loss_11.png?raw=true "11 layer model")   ![13 layer model](figures/trainVal_loss_13.png?raw=true "13 layer model")-->

<!--![11 layer model](figures/trainVal_loss_16.png?raw=true "16 layer model")   ![13 layer model](figures/trainVal_loss_13.png?raw=true "13 layer model")-->



<p>
  <img src="figures/trainVal_loss_11.png" width="600" title="11 layer model">
  <img src="figures/trainVal_loss_13.png" width="600" title="11 layer model">
</p>

<p>
  <img src="figures/trainVal_loss_16.png" width="600" title="11 layer model">
  <img src="figures/trainVal_loss_13.png" width="600" title="11 layer model">
</p>




<b>Accuracy:</b>
Coming soon

|layers Vs Dataset    | Trainset | Valset | Testset |
|-------------------|----------|--------|---------|
| 11 layers         | 0.4800       |  0.3591      |         |
| 13 layers         |          |        |         |
| 16 layers         |          |        |         |
| 16 layers, Cf:1x1 |          |        |         |
| 19 layers         |          |        |         |

<b>Loss:</b>
Coming soon

|layers Vs Dataset    | Trainset | Valset | Testset |
|-------------------|----------|--------|---------|
| 11 layers         |  2.03        |  2.75  |         |
| 13 layers         |          |        |         |
| 16 layers         |          |        |         |
| 16 layers, Cf:1x1 |          |        |         |
| 19 layers         |          |        |         |

### Usage

Very Deep Convolutional Networks for Large Scale Image Recognition
    
    [-h] [-t TRAIN] [-v VAL] [-b {64,128,256}] [-e {50,100,150}]
    [-d {11,13,16,19}] [-c11] [-es EARLY_STOPPING] [-i IMAGESIZE] [-lr LR]


 optional arguments:
 
      -h, --help            show this help message and exit
      -t TRAIN, --train TRAIN
                            required image dataset for training a model. It must
                            be in the data directory
      -v VAL, --val VAL     required image dataset for training a model. It must
                            be in the data directory
      -b {64,128,256}, --batchsize {64,128,256}
                            select number of samples to load from dataset
      -e {50,100,150}, --epochs {50,100,150}
      -d {11,13,16,19}, --depth {11,13,16,19}
                            depth of the deep learning model
      -c11, --conv1_1       setting it True will replace some of the 3x3 Conv
                            layers with 1x1 Conv layers in the 16 layer network
      -es EARLY_STOPPING, --early_stopping EARLY_STOPPING
                            early stopping is used to stop training of network, if
                            does not improve validation loss
      -i IMAGESIZE, --imagesize IMAGESIZE
                            it is used to resize the image pixels
      -lr LR, --lr LR       learning rate for an Adam optimizer


<b>Example usage:</b>

For training a model having a layer depth of 11:
            
    python train.py -d 11 -e 50 -es 5 -b 50 
           
For training a model having a layer depth of 16 along with smaller Conv filter 1x1:
        
    python train.py -d 16 -c11 -e 50 -es 5 -b 50 

