
from util import *
import os
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool
from config import *


def get_wavelist():
    train_dir = '../audio_train'
    test_dir = '../audio_test'
    waves_train = sorted(os.listdir(train_dir))
    waves_test = sorted(os.listdir(test_dir))
    print(len(waves_train)+len(waves_test))
    df_train = pd.DataFrame({'fname': waves_train})
    df_train['train0/test1'] = pd.DataFrame(0 for i in range(len(waves_train)))

    df_test = pd.DataFrame({'fname': waves_test})
    df_test['train0/test1'] = pd.DataFrame(1 for i in range(len(waves_test)))

    df = df_train.append(df_test)
    df.set_index('fname', inplace=True)
    df.to_csv('./wavelist.csv')


def wav_to_pickle(wavelist):
    sr = 22050

    df = pd.read_csv(wavelist)
    # print(df)
    for idx, item in df.iterrows():
        # print(item['fname'])
        if item['train0/test1'] == 0:
            file_path = os.path.join('../audio_train/', item['fname'])
        elif item['train0/test1'] == 1:
            file_path = os.path.join('../audio_test/', item['fname'])

        print(idx, file_path)
        data, _ = librosa.core.load(file_path, sr=sr, res_type='kaiser_best')
        p_name = os.path.join('../data-22050', os.path.splitext(item['fname'])[0] + '.pkl')
        # print(p_name)
        save_data(p_name, data)


def wav_to_pickle_parallel(wavelist):
    sr = 22050

    df = pd.read_csv(wavelist)
    # print(df)
    pool = Pool(10)
    pool.map(tsfm_wave, df.iterrows())


def wav_to_logmel(wavelist):

    df = pd.read_csv(wavelist)
    # print(df)
    pool = Pool(1)
    pool.map(tsfm_logmel, df.iterrows())


def tsfm_wave(row):
    sr = 22050
    item = row[1]
    if item['train0/test1'] == 0:
        file_path = os.path.join('../audio_train/', item['fname'])
    elif item['train0/test1'] == 1:
        file_path = os.path.join('../audio_test/', item['fname'])

    print(row[0], file_path)
    data, _ = librosa.core.load(file_path, sr=sr, res_type='kaiser_best')
    p_name = os.path.join('../data-22050-2', os.path.splitext(item['fname'])[0] + '.pkl')
    save_data(p_name, data)



def tsfm_logmel(row):

    item = row[1]
    p_name = os.path.join('../logmel_w40_s10_m64', os.path.splitext(item['fname'])[0] + '.pkl')
    if not os.path.exists(p_name):
        if item['train0/test1'] == 0:
            file_path = os.path.join('../audio_train/', item['fname'])
        elif item['train0/test1'] == 1:
            file_path = os.path.join('../audio_test/', item['fname'])

        print(row[0], file_path)
        data, _ = librosa.load(file_path, config.sampling_rate)
        melspec = librosa.feature.melspectrogram(data, config.sampling_rate,
                                                 n_fft=config.n_fft, hop_length=config.hop_length,
                                                 n_mels=config.n_mels)

        logmel = librosa.core.power_to_db(melspec)
        # p_name = os.path.join('../logmel_w40_s10_m64', os.path.splitext(item['fname'])[0] + '.pkl')
        # print(p_name)
        save_data(p_name, logmel)


if __name__ == '__main__':
    make_dirs()
    config = Config(sampling_rate=22050, n_mels=64, frame_weigth=40, frame_shift=10)
    # get_wavelist()
    wav_to_pickle_parallel('wavelist.csv')
    # wav_to_logmel('wavelist.csv')