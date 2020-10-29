from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import wave
import numpy as np

# train_path = './dataset/train/go/'
# file_name = '0a4e48db-fca9-4c48-a1cd-34be7053f371.wav'

# sample_rate, sample = wavfile.read(train_path + file_name)

def get_data(path):
  data = []

  for label in os.listdir(path):
    data_path = path + label + '/'
    for filename in os.listdir(data_path):
      data_name = data_path + filename
      sample_rate, sample = wavfile.read(data_name)
      data.append((sample_rate, sample, label))

  return data

train_data = get_data('./dataset/train/')

print(train_data[0])

plt.plot(range(len(train_data[0][1])), train_data[0][1])
plt.show()
