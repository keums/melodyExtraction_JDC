# -*- coding: utf-8 -*-

import numpy as np
import librosa
import matplotlib.pyplot as plt

def spec_extraction(file_name,win_size):
    print(file_name)

    x_test = []
    y, sr = librosa.load(file_name, sr=8000)

    S = librosa.core.stft(y, n_fft=1024, hop_length=80*1, win_length=1024)
    x_spec = np.abs(S)
    x_spec  = librosa.core.power_to_db(x_spec,ref=np.max)
    x_spec = x_spec.astype(np.float32)
    num_frames = x_spec.shape[1]

    # for padding
    padNum = num_frames % win_size
    if padNum != 0:
        len_pad = win_size - padNum
        padding_feature = np.zeros(shape=(513, len_pad))
        x_spec = np.concatenate((x_spec, padding_feature), axis=1)
        num_frames = num_frames + len_pad

    for j in range(0, num_frames, win_size):
        x_test_tmp = x_spec[:, range(j, j + win_size)].T
        x_test.append(x_test_tmp)
    x_test = np.array(x_test)

    # for normalization
    x_train_mean = np.load('./x_data_mean_total_31.npy')
    x_train_std = np.load('./x_data_std_total_31.npy')
    x_test = (x_test-x_train_mean)/(x_train_std+0.0001)
    x_test = x_test[:, :, :, np.newaxis]

    return x_test, x_spec
