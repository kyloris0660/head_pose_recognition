from keras.layers import *
import keras
from config import *
from densenet import denseblock, transition


def get_model():
    if MODEL_NAME == 'toy_model':
        return toy_model()
    if MODEL_NAME == 'Densenet_model':
        return Densenet_model()


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
    net.add(Dense(NUM_LABEL))

    return net


def Densenet_model(depth=22, nb_dense_block=3, num_filter=32, growing_rate=12, dropout_rate=0.2):
    model_input = Input((FEATURE_DIM_1, FEATURE_DIM_2, NUM_CHANNEL))
    nb_layers = int((depth - 4) / 3)
    x = Conv2D(num_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False,
               kernel_regularizer=keras.regularizers.l2(1E-4))(model_input)

    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis=-1,
                                  nb_layers=nb_layers, growth_rate=growing_rate,
                                  nb_filter=num_filter, dropout_rate=dropout_rate,
                                  weight_decay=1E-4)
        x = transition(x, nb_filter=num_filter, concat_axis=-1, dropout_rate=dropout_rate,
                       weight_decay=1E-4)

    x, nb_filter = denseblock(x, concat_axis=-1,
                              nb_layers=nb_layers, growth_rate=growing_rate,
                              nb_filter=num_filter, dropout_rate=dropout_rate,
                              weight_decay=1E-4)

    x = BatchNormalization(axis=-1,
                           gamma_regularizer=keras.regularizers.l2(1E-4),
                           beta_regularizer=keras.regularizers.l2(1E-4))(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling2D(data_format=K.image_data_format())(x)
    x = Dense(NUM_LABEL)(x)

    densenet = keras.models.Model(inputs=[model_input], outputs=[x])

    return densenet
