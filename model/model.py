from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from model.activations import Mish
__version__ = "1.0.0"
__author__ = "Willian Antunes"


class BrazilianIdModel:
    def __init__(self):
        super(BrazilianIdModel, self).__init__()
        self.model = self.get_model()

    def get_model(self):
        # Block 1
        inputs = Input(shape=(150, 150, 1))
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1))(inputs)
        x = Activation(Mish())(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = Activation(Mish())(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 3
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation(Mish())(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Block 4
        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation(Mish())(x)
        x = Dropout(.3)(x)
        outputs = Dense(units=8, activation='softmax')(x)

        return Model(inputs=[inputs], outputs=[outputs])

    def __getattr__(self, name):
        return getattr(self.model, name)
