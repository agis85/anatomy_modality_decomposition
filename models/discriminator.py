import sys

from keras import Input, Model
from keras.layers import LeakyReLU, Flatten, Dense, Conv2D
from keras.optimizers import Adam

from layers.spectralnorm import Spectral
from models.basenet import BaseNet

sys.path.append('../layers')


class Discriminator(BaseNet):
    '''
    LS-GAN Discriminator
    '''

    def __init__(self, conf):
        super(Discriminator, self).__init__(conf)

    def build(self):
        inp_shape         = self.conf.input_shape
        downsample_blocks = self.conf.downsample_blocks
        output            = self.conf.output
        name              = self.conf.name
        spectral          = self.conf.spectral
        f                 = self.conf.filters

        d_input = Input(inp_shape)
        l = conv2d(f, 4, 2, False, None, d_input)
        l = LeakyReLU(0.2)(l)

        for i in range(downsample_blocks):
            s = 1 if i == downsample_blocks - 1 else 2
            spectral_params = f * (2 ** i)
            l = self._downsample_block(l, f * 2 * (2 ** i), s, spectral, spectral_params)

        if output == '2D':
            spectral_params = f * (2 ** downsample_blocks) * 4 * 4
            l = conv2d(1, 4, 1, spectral, spectral_params, l)
        elif output == '1D':
            l = Flatten()(l)
            l = Dense(1, activation='linear')(l)

        self.model = Model(d_input, l, name=name)

    def _downsample_block(self, l0, f, stride, spectral, spectral_params, name=''):
        l = conv2d(f, 4, stride, spectral, spectral_params * 4 * 4, l0, name)
        return LeakyReLU(0.2)(l)

    def compile(self):
        assert self.model is not None, 'Model has not been built'
        self.model.compile(optimizer=Adam(lr=self.conf.lr, beta_1=0.5, decay=self.conf.decay), loss='mse')


def conv2d(filters, kernel, stride, spectral, spectral_params, l0, name=''):
    if spectral:
        l = Conv2D(filters, kernel, strides=stride, padding='same',
                   kernel_regularizer=Spectral(spectral_params, 10.), name=name)(l0)
    else:
        l = Conv2D(filters, kernel, strides=stride, padding='same', name=name)(l0)
    return l
