import librosa
import numpy as np
import os


# 读取数据
def get_data(path):
    data = []
    labels_names = []
    idx = 0
    
    for label in os.listdir(path):
        data_path = path + label + '/'
        labels_names.append(label)
        data.append([])
        for filename in os.listdir(data_path):
            data_name = data_path + filename
            sample, old_sr = librosa.load(data_name, sr=None)
            sr = 16000
            sample = librosa.resample(sample, old_sr, sr)
            data[idx].append((sample, sr))
        idx += 1
    return data, labels_names


# logmelspectrum
def preprocess_mel(data):
    sample, sr = data
    S = librosa.feature.melspectrogram(sample, sr=sr, n_mels=120)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S.resize(120, 32, 1)
    return log_S


# mfcc
def preprocess_mfcc(data):
    sample, sr = data
    S = librosa.feature.melspectrogram(sample, sr=sr, n_mels=120)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=40)
    delta2mfcc = librosa.feature.delta(mfcc, order=2)
    delta2mfcc = (delta2mfcc - np.mean(delta2mfcc)) / np.std(delta2mfcc)
    delta2mfcc.resize(40, 32, 1)
    return delta2mfcc
