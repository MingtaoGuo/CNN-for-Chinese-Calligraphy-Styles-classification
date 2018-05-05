# Simple-CNN-for-Chinese-character-classification
This code mainly solves the problem of style classification of Chinese calligraphy.

Indroduce
----------

For example, in China there are five famous fonts which are clerical script,standard script,semi-cursive,cursive script and seal script.
<div align=center><img src="https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/fonts/fivefont.jpg"/></div>
![fivefonts](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/fonts/fivefont.jpg)

There are four famous font of standard script that is ou yan liu zhao.

![fivefonts](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/fonts/fivefonts.jpg)![ouyanliuzhao](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/fonts/ouyanliuzhao.jpg)

Method
--------

We use a simple CNN to classify different style fonts,which include four convolution layers, we remove the fully connected layer, and add a new module that is called "Squeeze-and-Excitation"[1] ,this module is really useful. Meanwhile, in every layer, we add the batch normalization[2] to accelerate the rate of training.

Dataset
---------

In this repository, we provide a dataset of four fonts about ou yan liu and zhao, please see the ouyanliuzhaoData.mat file for detail.  
ouyanliuzhaoData.mat include 4 data :train train_label test test_label

train:2400 rows 4096 columns,it correspondes to images of 2400 64x64 images from reshaping

train_label:2400 rows 4 columns,one_hot coding

test:800 rows 4096 columns,it correspondes to images of 800 64x64 images from reshaping

test_label:800 rows 4 columns,one_hot coding

[1]Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks[J]. 2017.

[2]Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
