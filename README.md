## freesound-audio-tagging

### Requirments:

python 3.6

pytorch 0.4.0

cuda 9.1

librosa 0.5.1

torchvision 0.2.1


### Data

From Kaggle competition https://www.kaggle.com/c/freesound-audio-tagging/data


### Run:

实例化 Config 类, 设置参数.

data_transform.py 用来转化训练数据.
例如将.wav文件转换为numpy格式存储,重采样,提取logmel特征,提取MFCC特征等.

train_on_logmel.py
利用logmel特征训练. (利用MFCC特征训练也是这个文件,需要更改config中的输入特征,
data_dir)

train_on_wave.py
利用波形特征训练.
(与train_on_logmel.py的主要区别在于DataLoader类和优化器optimizer的选取)


### 一些小问题

测试集中有三个文件是空的,处理数据的时候需要注意:

b39975f5.wav

0b0427e2.wav

6ea0099f.wav

目前的做法是将数据填0.
