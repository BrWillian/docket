import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class Mish(Layer):
    '''
    @Author Willian Antunes

    Referência:
    @article{misra2019mish,
      title={Mish: A self regularized non-monotonic neural activation function},
      author={Misra, Diganta},
      journal={arXiv preprint arXiv:1908.08681},
      year={2019}
    }
    '''
    def __init__(self, name=None):
        super(Mish, self).__init__(name=name)

    @tf.function
    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape
