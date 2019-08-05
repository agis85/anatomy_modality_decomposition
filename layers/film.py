import keras.backend as K
from keras.engine import Layer


class FiLM(Layer):
    '''
    The FiLM Normalization of E. Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer, AAAI, 2018.

    Usage:
        h = FiLM()([h, gamma, beta])

    where:
        h is the multi channel image with shape (?, H, W, C)
        gamma has shape (?, C), and scales the channels (values in range -inf,inf)
        beta has shape (?, C), and offsets the channels (values in range -inf,inf)
    '''

    def __init__(self, **kwargs):
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FiLM, self).build(input_shape)

    def call(self, x, **kwargs):
        x, gamma, beta = x

        print('FILM: ', K.int_shape(x), K.int_shape(gamma))

        gamma = K.tile(K.reshape(gamma, (K.shape(gamma)[0], 1, 1, K.shape(gamma)[-1])),
                       (1, K.shape(x)[1], K.shape(x)[2], 1))
        beta = K.tile(K.reshape(beta, (K.shape(beta)[0], 1, 1, K.shape(beta)[-1])),
                      (1, K.shape(x)[1], K.shape(x)[2], 1))

        return x * gamma + beta

    def compute_output_shape(self, input_shape):
        return input_shape