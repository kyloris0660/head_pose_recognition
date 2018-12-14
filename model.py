from keras.layers import *
import keras
from config import *


def get_model():
    if MODEL_NAME == 'toy_model':
        return toy_model()


def toy_model():
    net = keras.models.Sequential()
    net.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu',
                   input_shape=(FEATURE_DIM_1, FEATURE_DIM_2, NUM_CHANNEL)))
    net.add(BatchNormalization())
    net.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())

    net.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())
    net.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())

    net.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())
    net.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())

    net.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())
    net.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(BatchNormalization())

    # net.add(Flatten())
    net.add(GlobalAveragePooling2D())
    net.add(Dense(64, activation='relu'))
    net.add(Dense(NUM_LABEL, activation='tanh'))

    net.compile(loss='mean_squared_error',metrics=['mse'],
                optimizer=keras.optimizers.Adadelta())
    return net