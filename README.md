# CNN-for-Chinese-Calligraphy-Styles-classification
This code mainly solves the problem of style classification of Chinese calligraphy.

The paper has published: [A novel CNN structure for fine-grained classification of Chinesecalligraphy styles](https://rdcu.be/bxS19)

Introduce
----------

![fivefonts](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/fivefonts.jpg)

There are four famous styles in standard font that is ou, yan, liu, zhao.

![fourstyles](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/fourstyles.jpg)

Methods
---------
We use Squeeze-and-Excitation block and haar-wavelet block to improve the performance of our network.

### The method of Squeeze-and-Excitation block:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/seblock.jpg)
### The method of haar-wavelet block:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/haarwavelet.jpg)
### The architecture of our model:
![SEBLOCK](https://github.com/MingtaoGuo/Simple-CNN-for-Chinese-character-classification/raw/master/IMGS/network.jpg)
