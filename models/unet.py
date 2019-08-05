from keras import Input, Model
from keras.layers import Concatenate, Conv2D, MaxPooling2D, LeakyReLU, Add, Activation, UpSampling2D, \
    BatchNormalization, Lambda
from keras_contrib.layers import InstanceNormalization

from models.basenet import BaseNet
import logging
log = logging.getLogger('unet')


class UNet(BaseNet):
    """
    UNet Implementation of 4 downsampling and 4 upsampling blocks.
    Each block has 2 convolutions, batch normalisation and relu.
    The number of filters for the 1st layer is 64 and at every block, this is doubled. Each upsampling block halves the
    number of filters.
    """
    def __init__(self, conf):
        """
        Constructor.
        :param conf: the configuration object
        """
        super(UNet, self).__init__(conf)
        self.input_shape  = conf.input_shape
        self.residual     = conf.residual
        self.out_channels = conf.out_channels
        self.normalise    = conf.normalise
        self.f            = conf.filters
        self.downsample   = conf.downsample
        assert self.downsample > 0, 'Unet downsample must be over 0.'

    def build(self):
        """
        Build the model
        """
        self.input = Input(shape=self.input_shape)
        l = self.unet_downsample(self.input, self.normalise)
        self.unet_bottleneck(l, self.normalise)
        l = self.unet_upsample(self.bottleneck, self.normalise)
        out = self.out(l)

        self.model = Model(inputs=self.input, outputs=out)
        self.model.summary(print_fn=log.info)
        self.load_models()

    def unet_downsample(self, inp, normalise):
        """
        Build downsampling path
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            last layer of the downsampling path
        """
        self.d_l0 = conv_block(inp, self.f, normalise, self.residual)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l0)

        if self.downsample > 1:
            self.d_l1 = conv_block(l, self.f * 2, normalise, self.residual)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l1)

        if self.downsample > 2:
            self.d_l2 = conv_block(l, self.f * 4, normalise, self.residual)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l2)

        if self.downsample > 3:
            self.d_l3 = conv_block(l, self.f * 8, normalise)
            l = MaxPooling2D(pool_size=(2, 2))(self.d_l3)
        return l

    def unet_bottleneck(self, l, normalise, name=''):
        """
        Build bottleneck layers
        :param inp:         input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :param name:        name of the layer
        """
        flt = self.f * 2
        if self.downsample > 1:
            flt *= 2
        if self.downsample > 2:
            flt *= 2
        if self.downsample > 3:
            flt *= 2
        self.bottleneck = conv_block(l, flt, normalise, self.residual, name)

    def unet_upsample(self, l, normalise):
        """
        Build upsampling path
        :param l:           the input layer
        :param normalise:   normalise type. Can be one of [batch, instance, None]
        :return:            the last layer of the upsampling path
        """
        if self.downsample > 3:
            l = upsample_block(l, self.f * 8, normalise, activation='linear')
            l = Concatenate()([l, self.d_l3])
            l = conv_block(l, self.f * 8, normalise, self.residual)

        if self.downsample > 2:
            l = upsample_block(l, self.f * 4, normalise, activation='linear')
            l = Concatenate()([l, self.d_l2])
            l = conv_block(l, self.f * 4, normalise, self.residual)

        if self.downsample > 1:
            l = upsample_block(l, self.f * 2, normalise, activation='linear')
            l = Concatenate()([l, self.d_l1])
            l = conv_block(l, self.f * 2, normalise, self.residual)

        if self.downsample > 0:
            l = upsample_block(l, self.f, normalise, activation='linear')
            l = Concatenate()([l, self.d_l0])
            l = conv_block(l, self.f, normalise, self.residual)

        return l

    def out(self, l, out_activ=None):
        """
        Build ouput layer
        :param l: last layer from the upsampling path
        :return:  the final segmentation layer
        """
        if out_activ is None:
            out_activ = 'sigmoid' if self.out_channels == 1 else 'softmax'
        return Conv2D(self.out_channels, 1, activation=out_activ)(l)


def conv_block(l0, f, norm_name, residual=False, name=''):
    """
    Convolutional block
    :param l0:        the input layer
    :param f:         number of feature maps
    :param residual:  True/False to define residual connections
    :return:          the last layer of the convolutional block
    """
    l = Conv2D(f, 3, strides=1, padding='same')(l0)
    l = normalise(norm_name)(l)
    l = Activation('relu')(l)
    l = Conv2D(f, 3, strides=1, padding='same')(l)
    l = normalise(norm_name)(l)
    if residual:
        Activation('relu')(l)
        return Add(name=name)([l0, l])
    return Activation('relu', name=name)(l)


def upsample_block(l0, f, norm_name, activation='relu'):
    """
    Upsampling block.
    :param l0:          input layer
    :param f:           number of feature maps
    :param activation:  activation name
    :return:            the last layer of the upsampling block
    """
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(f, 3, padding='same')(l)
    l = normalise(norm_name)(l)

    if activation == 'leakyrelu':
        return LeakyReLU()(l)
    else:
        return Activation(activation)(l)


def normalise(norm=None, **kwargs):
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x : x)