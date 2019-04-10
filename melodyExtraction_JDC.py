
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:04:22 2019

@author: keums
"""

import os
import click
import librosa
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from model import *
from featureExtraction import *


class Options(object):
    def __init__(self):
        self.num_spec = 513
        self.input_size = 31  # 115
        self.batch_size = 64  # 64

        self.use_multi_gpu = 1
        self.resolution = 16
        self.thres_v = 0.5
        self.figureON = False  # True


options = Options()


@click.command()
@click.argument('file_name', type=click.Path(exists=True))
@click.argument('outfile', type=click.Path())
def main(file_name, outfile):
    pitch_range = np.arange(
        38, 83 + 1.0/options.resolution, 1.0/options.resolution)
    pitch_range = np.concatenate([np.zeros(1), pitch_range])

    '''  Extracting features '''
    X_test, X_spec = spec_extraction(
        file_name=file_name, win_size=options.input_size)

    '''  Prediction of melody '''
    model = melody_ResNet_joint_add(options)
    model.load_weights('./weights/ResNet_joint_add_L(CE_G).hdf5')
    y_predict = model.predict(X_test, batch_size=options.batch_size, verbose=1)

    num_total = y_predict[0].shape[0] * y_predict[0].shape[1]
    # y_predict_v = np.reshape(y_predict[1], (num_total, 2))

    est_pitch = np.zeros(num_total)
    index_predict = np.zeros(num_total)

    y_predict = np.reshape(
        y_predict[0], (num_total, y_predict[0].shape[2]))  # origin
    for i in range(y_predict.shape[0]):
        index_predict[i] = np.argmax(y_predict[i, :])
        pitch_MIDI = pitch_range[np.int32(index_predict[i])]
        if pitch_MIDI >= 38 and pitch_MIDI <= 83:
            est_pitch[i] = 2 ** ((pitch_MIDI - 69) / 12.) * 440

    est_pitch = medfilt(est_pitch, 5)
    writeMel(est_pitch, outfile)

    ''' Plot '''
    if options.figureON == True:
        start = 2000
        end = 7000
        fig = plt.figure()
        plt.imshow(X_spec[:, start:end], origin='lower')
        plt.plot(est_pitch[start:end], 'r', linewidth=0.5)
        fig.tight_layout()
        plt.show()

        # plt.savefig('test.pdf', bbox_inches='tight')


def writeMel(pitches, outfile):
    times = librosa.frames_to_time(
        np.arange(len(pitches)), sr=8000, hop_length=80*1)
    np.savetxt(outfile, np.array([times, pitches]).T, fmt=[
               '%.3f', '%.4f'], delimiter='\t')


if __name__ == '__main__':
    options = Options()
    # file_name = '/Project/dataset/musdb18/test/Cristina Vane - So Easy.stem.mp4'
    main()
