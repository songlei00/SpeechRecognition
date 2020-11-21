from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras import initializers


def vgg_net(input_shape, n_outputs):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=initializers.he_uniform())(x)
    x = Dense(256, activation='relu', kernel_initializer=initializers.he_uniform())(x)
    outputs = Dense(n_outputs, activation='softmax', kernel_initializer=initializers.he_uniform())(x)
    
    return keras.Model(inputs, outputs)


if __name__ == '__main__':
    vgg = vgg_net((40, 32, 1), 6)
    keras.utils.plot_model(vgg, 'vgg.png', show_shapes=True)