import time

from util import *
from config import *
import os
from network import *
import numpy as np
from torchvision import models
from matplotlib import pyplot as plt

config = Config()
# attrs = '\n'.join('%s:%s'%item for item in vars(config).items())
#
#
#
# s = 'resnet101_m'
# model = run_method_by_string(s)()
# print(model)

# logging = create_logging('../log', filemode='w')
#
# logging.info(os.path.abspath(__file__))
# logging.info(attrs)


# logmel = np.ones((2,3))
# print(logmel)
# logmel = np.pad(logmel, ((0, 0),(0, 4)), "constant")
# print(logmel)

# data, _ = librosa.load('../audio_test/0b0427e2.wav', config.sampling_rate)
# melspec = librosa.feature.melspectrogram(data, config.sampling_rate,
#                                          n_fft=config.n_fft, hop_length=config.hop_length,
#                                          n_mels=config.n_mels)
# logmel = librosa.core.power_to_db(melspec)
# print(data.shape)

# logmel = np.ones((3, 64, 100))
# print(logmel.shape)
# logmel = np.pad(logmel, ((0, 0), (0, 0), (0, 2)), "constant")
# print(logmel.shape)
# print(logmel)

# from sklearn.model_selection import StratifiedKFold
# X = np.array([0,1,2,3,4])
# y = np.array([0,0,0,0,0])
# skf = StratifiedKFold(n_splits=5)
# print(skf.get_n_splits(X, y))
# for train, label in skf.split(X, y):
#     print(train, label)

a = np.random.beta(3, 3, 2)
# b = np.random.permutation(5)
print(a)

# plt.subplot(nrows=2, ncols=1, 2)

