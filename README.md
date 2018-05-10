#freesound-audio-tagging

### Requirments:

python 3.6

pytorch 0.4.0

cuda 9.1

librosa 0.5.1

torchvision 0.2.1

### Run:

修改config.py

python data_transform.py

python train_on_logmel.py (利用logmel特征训练)


### 还可以做的:

清洗数据,数据集中有一些数据是空的.

提前将logmel特征提取出来,加速训练.

利用一维卷积从wave提特征.

