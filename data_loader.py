import numpy as np
np.random.seed(1001)

import librosa
import os
import pandas as pd
from config import Config
from util import *

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import time

class Freesound(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        # dict for mapping class names into indices. can be obtained by
        # {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
        # self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28,
        #                 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7, 'Computer_keyboard': 8, 'Cough': 17,
        #                 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14,
        #                 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26,
        #                 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5,
        #                 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24,
        #                 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18,
        #                 'Trumpet': 2, 'Violin_or_fiddle': 39, 'Writing': 11}

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]


    def __getitem__(self, idx):

        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.data_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        data = data[np.newaxis, :]

        if self.mode is "train":
            # label_name = self.frame["label"][idx]
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data


    def _random_selection(self, file_path):

        input_length = self.config.audio_length
        # Read and Resample the audio
        data = load_data(file_path)

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data


class Freesound_logmel(Dataset):
    def __init__(self, config, frame, mode, transform=None):
        self.config = config
        self.frame = frame
        # dict for mapping class names into indices. can be obtained by
        # {cls_name:i for i, cls_name in enumerate(csv_file["label"].unique())}
        # self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28,
        #                 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7, 'Computer_keyboard': 8, 'Cough': 17,
        #                 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14,
        #                 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26,
        #                 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5,
        #                 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24,
        #                 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18,
        #                 'Trumpet': 2, 'Violin_or_fiddle': 39, 'Writing': 11}
        # self.classes = {cls_name:i for i, cls_name in enumerate(config.labels)}

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.frame.shape[0]


    def __getitem__(self, idx):

        filename = os.path.splitext(self.frame["fname"][idx])[0] + '.pkl'

        file_path = os.path.join(self.config.data_dir, filename)

        # Read and Resample the audio
        data = self._random_selection(file_path)

        if self.transform is not None:
            data = self.transform(data)

        # data = data[np.newaxis, :]

        if self.mode is "train":
            # label_name = self.frame["label"][idx]
            label_idx = self.frame["label_idx"][idx]
            return data, label_idx
        if self.mode is "test":
            return data


    def _random_selection(self, file_path):

        input_frame_length = int(self.config.audio_duration * 1000 / self.config.frame_shift)
        # Read the logmel pkl
        logmel = load_data(file_path)

        # Random offset / Padding
        if logmel.shape[2] > input_frame_length:
            max_offset = logmel.shape[2] - input_frame_length
            offset = np.random.randint(max_offset)
            data = logmel[:, :, offset:(input_frame_length + offset)]
        else:
            if input_frame_length > logmel.shape[2]:
                max_offset = input_frame_length - logmel.shape[2]
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(logmel, ((0, 0), (0, 0), (offset, input_frame_length - logmel.shape[2] - offset)), "constant")
        return data

#
# class ToLogMel(object):
#     def __init__(self, config):
#         self.config = config
#
#     def __call__(self, data):
#         melspec = librosa.feature.melspectrogram(data, self.config.sampling_rate,
#                                                  n_fft=2048, hop_length=75,
#                                                  n_mels=self.config.n_mels)  # (64, 442)
#         logmel = librosa.logamplitude(melspec)[:,:441]  # (64, 441)
#         # logmel = np.ones((64, 441))
#         return logmel


class ToTensor(object):
    """#{{{
    convert ndarrays in sample to Tensors.
„
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)

    """
    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data


if __name__ == "__main__":
    # config = Config(sampling_rate=44100, audio_duration=1.5, data_dir="../data-22050")
    config = Config(sampling_rate=22050, audio_duration=1.5, data_dir="../mfcc+delta_w80_s10_m64")
    DEBUG = True

    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../sample_submission.csv')

    LABELS = config.labels
    # LABELS = list(train.label.unique())
    # ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Knock',
    # 'Gunshot_or_gunfire', 'Clarinet', 'Computer_keyboard', 'Keys_jangling',
    # 'Snare_drum', 'Writing', 'Laughter', 'Tearing', 'Fart', 'Oboe', 'Flute',
    # 'Cough', 'Telephone', 'Bark', 'Chime', 'Bass_drum', 'Bus', 'Squeak',
    # 'Scissors', 'Harmonica', 'Gong', 'Microwave_oven', 'Burping_or_eructation',
    # 'Double_bass', 'Shatter', 'Fireworks', 'Tambourine', 'Cowbell',
    # 'Electric_piano', 'Meow', 'Drawer_open_or_close', 'Applause', 'Acoustic_guitar',
    # 'Violin_or_fiddle', 'Finger_snapping']

    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname")
    test.set_index("fname")
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])

    if DEBUG:
        train = train[:2000]
        test = test[:2000]

    skf = StratifiedKFold(n_splits=config.n_folds)

    for foldNum, (train_split, val_split) in enumerate(skf.split(train, train.label_idx)):
        # print("TRAIN:", train_split, "VAL:", val_split)
        # train_set = train.iloc[train_split]
        # train_set = train_set.reset_index(drop=True)
        # val_set = train.iloc[val_split]
        # val_set = val_set.reset_index(drop=True)
        # print(len(train_set), len(val_set))
        #
        # trainSet = Freesound(config=config, frame=train_set,
        #                      transform=transforms.Compose([
        #                          ToLogMel(config),
        #                          ToTensor()
        #                      ]),
        #                      mode="train")
        # train_loader = DataLoader(trainSet, batch_size=5, shuffle=False, num_workers=1)

        # for i, (input, target) in enumerate(train_loader):
        #     print(i)
        #     print(input)
        #     print(input.size())
        #     print(target)
        #     break


        #---------test logmel loader------------
        # test_set = pd.read_csv('../sample_submission.csv')
        # testSet = Freesound_logmel(config=config, frame=test_set,
        #                     transform=transforms.Compose([ToTensor()]),
        #                     mode="test")
        # test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=1)
        #
        # for i, input in enumerate(test_loader):
        #     print(i)
        #     print(input)
        #     print(input.size())
        #     print(input.type())
        #     break

        # ---------test logmel loader------------
        test_set = pd.read_csv('../sample_submission.csv')
        testSet = Freesound_logmel(config=config, frame=test_set,
                                   # transform=transforms.Compose([ToTensor()]),
                                   mode="test")
        test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=4)
        print(len(test_loader))
        for i, input in enumerate(test_loader):
            if i == len((test_loader))-1:
                # print(i)
                print(input)
            # print(input)
                print(input.size())
            # print(input.type())
            # break