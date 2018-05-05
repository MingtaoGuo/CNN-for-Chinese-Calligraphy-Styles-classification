# Simple-CNN-for-Chinese-character-classification
This code mainly solves the problem of style classification of Chinese calligraphy.

Indroduce

For example, in China there are five famous fonts which are clerical script,standard script,semi-cursive,cursive script and seal script.
There are four famous font of standard script that is ou yan liu zhao.

Method

We use a simple CNN to classify different style fonts,which include four convolution layers, we remove the fully connected layer, and add a new module that is called "Squeeze-and-Excitation"[1] ,this module is really useful. Meanwhile, in every layer, we add the batch normalization[2] to accelerate the rate of training.

[1]Hu J, Shen L, Sun G. Squeeze-and-Excitation Networks[J]. 2017.

[2]Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
