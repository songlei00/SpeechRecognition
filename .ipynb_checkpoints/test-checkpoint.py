import numpy as np
import argparse
from tensorflow import keras

from tools import get_data, preprocess_mel, preprocess_mfcc
from densenet import DenseNet
from vggnet19 import vgg_net
from resnet50 import ResNet50
from warmup_cosdecay import WarmUpCosineDecayScheduler


# 参数解析
parser = argparse.ArgumentParser()

parser.add_argument("--test_path", default='dataset/test/', type=str, required=False,
                        help="The dir for test data")
args = parser.parse_args()


# 读取数据
raw_test_data, test_label_names = get_data(args.test_path)
label_dit = {'go': 0, 'left': 1, 'off': 0, 'on': 0, 'right': 2, 'stop': 3}

# mfcc 
delta_mfcc_test_data = []
delta_mfcc_test_labels = []
for i in range(len(raw_test_data)):
    delta_mfcc_test_data.extend(list(map(preprocess_mfcc, raw_test_data[i])))
    delta_mfcc_test_labels.extend(np.ones(len(raw_test_data[i])).astype(np.int32) * label_dit[test_label_names[i]])
    
delta_mfcc_test_data = np.array(delta_mfcc_test_data)
delta_mfcc_test_labels = np.array(delta_mfcc_test_labels)

# mel
mel_test_data = []
mel_test_labels = []
for i in range(len(raw_test_data)):
    mel_test_data.extend(list(map(preprocess_mel, raw_test_data[i])))
    mel_test_labels.extend(np.ones(len(raw_test_data[i])).astype(np.int32) * label_dit[test_label_names[i]])

mel_test_data = np.array(mel_test_data)
mel_test_labels = np.array(mel_test_labels)


# vgg
vgg = vgg_net((40, 32, 1), 4)
optimizer_vgg = keras.optimizers.Adam()
vgg.compile(
    optimizer=optimizer_vgg,
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
vgg.load_weights('vgg_net_weight/vgg19_net')


# resnet
resnet = ResNet50(input_shape=(120, 32, 1), classes=4)
optimizer_resnet = keras.optimizers.Adam()
resnet.compile(
    optimizer=optimizer_resnet,
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
resnet.load_weights('resnet_weight/resnet')


# densenet
densenet = DenseNet(4, (120, 32, 1), dropout_rate=0.4)
optimizer_densenet = keras.optimizers.Adam()
densenet.compile(
    optimizer=optimizer_densenet,
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)
densenet.load_weights('densenet/densenet')


# evaluate
acc = sum(
    (vgg.predict(delta_mfcc_test_data) + 
     resnet.predict(mel_test_data) + 
     densenet.predict(mel_test_data)).argmax(axis=1) == mel_test_labels
) / len(mel_test_labels)

print('accuracy:', acc)
