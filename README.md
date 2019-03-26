# Simple-CNN-for-Chinese-character-classification
This code mainly solves the problem of style classification of Chinese calligraphy.

Introduce
----------

For example, in China there are five famous fonts which are clerical script,standard script,semi-cursive,cursive script and seal script.
<div align=center><img src="https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/fivefonts.jpg"/></div>

There are four famous styles in standard font that is ou, yan, liu, zhao.

![fivefonts](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/fourstyles.jpg)

Methods
---------
We use Squeeze-and-Excitation block and haar-wavelet block to improve the performance of our network.

### The method of Squeeze-and-Excitation block:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/seblock.jpg)
### The method of haar-wavelet block:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/haarwavelet.jpg)
### The architecture of our model:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/network.jpg)
