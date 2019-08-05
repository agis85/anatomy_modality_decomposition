
import logging
import os

import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Lambda, UpSampling2D, \
    Concatenate, BatchNormalization, Reshape, Add
from keras.optimizers import Adam
from keras_contrib.layers import InstanceNormalization

import costs
from layers.film import FiLM
from layers.rounding import Rounding
from models.basenet import BaseNet
from models.discriminator import Discriminator
from models.unet import UNet
from utils.sdnet_utils import sampling, make_trainable, get_net

log = logging.getLogger('sdnet')


class SDNet(BaseNet):
    """
    The SDNet model builder.
    """
    def __init__(self, conf):
        """
        SDNet constructor
        :param conf: configuration object
        """
        super(SDNet, self).__init__(conf)

        self.w_kl    = K.variable(self.conf.w_kl)
        self.w_rec_X = K.variable(self.conf.w_rec_X)
        self.w_sup_M = K.variable(self.conf.w_sup_M)
        self.w_rec_Z = K.variable(self.conf.w_rec_Z)

        self.D_Mask       = None  # Mask Discriminator
        self.Enc_Anatomy  = None  # Anatomy Encoder
        self.Enc_Modality = None  # Modality Encoder
        self.Segmentor    = None  # Segmentor
        self.Decoder      = None  # Decoder
        self.Ind_Rep      = None  # Independence Representation Trainer
        self.G_trainer    = None  # Trainer when having unlabelled data
        self.G_supervised_trainer = None  # Trainer when using data with labels.
        self.Z_Regressor  = None  # Trainer for reconstructing a sampled Z
        self.D_trainer    = None  # Trainer for mask discriminator

    def build(self):
        """
        Build the model's components
        """
        self.build_mask_discriminator()
        self.build_generators()
        self.load_models()

    def load_models(self):
        """
        Load weights from saved model files
        """
        if os.path.exists(self.conf.folder + '/G_trainer'):
            log.info('Loading trained models from file')

            self.G_trainer.load_weights(self.conf.folder + '/G_trainer')
            self.G_supervised_trainer.load_weights(self.conf.folder + '/G_supervised_trainer')
            self.D_trainer.load_weights(self.conf.folder + '/D_trainer')

            self.Enc_Anatomy  = get_net(self.G_trainer, 'Enc_Anatomy')
            self.Enc_Modality = get_net(self.G_trainer, 'Enc_Modality')
            self.Segmentor    = get_net(self.G_trainer, 'Segmentor')
            self.Decoder      = get_net(self.G_trainer, 'Reconstructor')
            self.D_Mask       = get_net(self.D_trainer, 'D_Mask')
            self.build_z_regressor()

    def save_models(self, postfix=''):
        """
        Save model weights in files.
        """
        log.debug('Saving trained models')
        self.G_trainer.save_weights(self.conf.folder + '/G_trainer' + postfix)
        self.G_supervised_trainer.save_weights(self.conf.folder + '/G_supervised_trainer' + postfix)
        self.D_trainer.save_weights(self.conf.folder + '/D_trainer' + postfix)
        if self.Ind_Rep is not None:
            self.Ind_Rep.save_weights(self.conf.folder + '/Ind_Rep' + postfix)

    def build_mask_discriminator(self):
        """
        Build a Keras model for training a mask discriminator.
        """
        # Build a discriminator for masks.
        D = Discriminator(self.conf.d_mask_params)
        D.build()
        log.info('Mask Discriminator D_M')
        D.model.summary(print_fn=log.info)
        self.D_Mask = D.model

        real_M = Input(self.conf.d_mask_params.input_shape)

        fake_M = Input(self.conf.d_mask_params.input_shape)
        real = self.D_Mask(real_M)
        fake = self.D_Mask(fake_M)
        self.D_trainer = Model([real_M, fake_M], [real, fake], name='D_trainer')
        self.D_trainer.compile(Adam(lr=self.conf.d_mask_params.lr, beta_1=0.5, decay=self.conf.d_mask_params.decay),
                               loss='mse')
        self.D_trainer.summary(print_fn=log.info)

    def build_generators(self):
        """
        Build encoders, segmentor, decoder and training models.
        """
        assert self.D_Mask is not None, 'Discriminator has not been built yet'
        make_trainable(self.D_trainer, False)

        self.build_anatomy_encoder()
        self.build_modality_encoder()
        self.build_segmentor()
        self.build_decoder()

        self.build_unsupervised_trainer()  # build standard gan for data with no labels
        self.build_supervised_trainer()
        self.build_z_regressor()

    def build_anatomy_encoder(self):
        """
        Build an encoder to extract anatomical information from the image.
        """
        # Manually build UNet to add Rounding as a last layer
        spatial_encoder = UNet(self.conf.anatomy_encoder_params)
        spatial_encoder.input = Input(shape=self.conf.input_shape)
        l1 = spatial_encoder.unet_downsample(spatial_encoder.input, spatial_encoder.normalise)
        spatial_encoder.unet_bottleneck(l1, spatial_encoder.normalise)
        l2 = spatial_encoder.unet_upsample(spatial_encoder.bottleneck, spatial_encoder.normalise)
        anatomy = spatial_encoder.out(l2, out_activ='softmax')
        if self.conf.rounding == 'encoder':
            anatomy = Rounding()(anatomy)

        self.Enc_Anatomy = Model(inputs=spatial_encoder.input, outputs=anatomy, name='Enc_Anatomy')
        log.info('Enc_Anatomy')
        self.Enc_Anatomy.summary(print_fn=log.info)

    def build_modality_encoder(self):
        """
        Build an encoder to extract intensity information from the image.
        """
        anatomy = Input(self.Enc_Anatomy.output_shape[1:])

        image = Input(self.conf.input_shape)

        l = Concatenate(axis=-1)([anatomy, image])
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(16, 3, strides=2)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Flatten()(l)
        l = Dense(32)(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)

        z_mean = Dense(self.conf.num_z, name='z_mean')(l)
        z_log_var = Dense(self.conf.num_z, name='z_log_var')(l)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, name='z')([z_mean, z_log_var])
        divergence = Lambda(costs.kl, name='divergence')([z_mean, z_log_var])

        self.Enc_Modality = Model(inputs=[anatomy, image], outputs=[z, divergence], name='Enc_Modality')
        log.info('Enc_Modality')
        self.Enc_Modality.summary(print_fn=log.info)

    def build_segmentor(self):
        """
        Build a segmentation network that converts anatomical maps to segmentation masks.
        """
        inp = Input(self.Enc_Anatomy.output_shape[1:])
        l = Conv2D(64, 3, strides=1, padding='same')(inp)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)
        l = Conv2D(64, 3, strides=1, padding='same')(l)
        l = BatchNormalization()(l)
        l = LeakyReLU()(l)

        conv_channels = self.loader.num_masks + 1  # +1 for softmax
        output = Conv2D(conv_channels, 1, padding='same', activation='softmax')(l)
        output = Lambda(lambda x: x[..., 0:conv_channels - 1])(output)

        self.Segmentor = Model(inputs=inp, outputs=output, name='Segmentor')
        log.info('Segmentor')
        self.Segmentor.summary(print_fn=log.info)

    def build_decoder(self):
        """
        Build a decoder that generates an image by combining an anatomical and a modality
        representation.
        """
        spatial_shape = tuple(self.conf.input_shape[:-1]) + (self.conf.num_mask_channels,)
        spatial_input = Input(shape=spatial_shape)

        resd_input = Input((self.conf.num_z,))  # (batch_size, 16)
        l1 = self._film_layer(spatial_input, resd_input)
        l2 = self._film_layer(l1, resd_input)
        l3 = self._film_layer(l2, resd_input)
        l4 = self._film_layer(l3, resd_input)

        l = Conv2D(1, 3, activation='tanh', padding='same')(l4)
        log.info('Reconstructor')
        self.Decoder = Model(inputs=[spatial_input, resd_input], outputs=l, name='Reconstructor')
        self.Decoder.summary(print_fn=log.info)

    def _film_pred(self, z, num_chn):
        """
        Given a z-sample, predict gamma and beta to apply FiLM.
        :param z:           a modality sample
        :param num_chn:     number of channels of the spatial feature maps
        :return:            the FiLM parameters
        """
        film_pred = Dense(num_chn)(z)
        film_pred = LeakyReLU()(film_pred)
        film_pred = Dense(num_chn)(film_pred)
        gamma = Lambda(lambda x: x[:, :int(num_chn/2)])(film_pred)
        beta  = Lambda(lambda x: x[:, int(num_chn/2):])(film_pred)
        return gamma, beta

    def _film_layer(self, spatial_input, resd_input):
        """
        A FiLM layer. Modulates the spatial input by the residual input.
        :param spatial_input:   the spatial features of the anatomy
        :param resd_input:      the modality features
        :return:                a modulated anatomy
        """
        l1 = Conv2D(self.conf.num_mask_channels, 3, padding='same')(spatial_input)
        l1 = LeakyReLU()(l1)

        l2 = Conv2D(self.conf.num_mask_channels, 3, strides=1, padding='same')(l1)
        gamma_l2, beta_l2 = self._film_pred(resd_input, 2 * self.conf.num_mask_channels)
        l2 = FiLM()([l2, gamma_l2, beta_l2])
        l2 = LeakyReLU()(l2)

        l = Add()([l1, l2])
        return l

    def build_unsupervised_trainer(self):
        """
        Model for training SDNet without labels using the mask discriminator and reconstruction cost.
        """
        # inputs
        real_X = Input(shape=self.conf.input_shape)
        fake_S = self.Enc_Anatomy(real_X)
        fake_Z, divergence = self.Enc_Modality([fake_S, real_X])

        # X -> S, Z
        fake_M = self.Segmentor(fake_S)
        fake_M = Lambda(lambda x : x[..., 0:self.D_Mask.input_shape[-1]])(fake_M)
        adv_M = self.D_Mask(fake_M)

        # S, Z -> X'
        rec_X = self.Decoder([fake_S, fake_Z])

        self.G_trainer = Model(inputs=real_X, outputs=[adv_M, rec_X, divergence])
        log.info('Unsupervised trainer')
        self.G_trainer.summary(print_fn=log.info)

    def build_supervised_trainer(self):
        """
        Model for training SDNet given labelled images. In addition to the unsupervised trainer, a direct segmentation
        cost is also minimised.
        """
        real_X = Input(self.conf.input_shape)
        fake_S = self.Enc_Anatomy(real_X)
        fake_Z, divergence = self.Enc_Modality([fake_S, real_X])
        fake_M = self.Segmentor(fake_S)
        rec_X = self.Decoder([fake_S, fake_Z])

        self.G_supervised_trainer = Model(inputs=real_X, outputs=[fake_M, rec_X, divergence])
        log.info('Supervised trainer')
        self.G_supervised_trainer.summary(print_fn=log.info)

    def build_z_regressor(self):
        if self.conf.w_rec_Z == 0:
            return

        sample_S = Input(self.Enc_Anatomy.output_shape[1:])
        sample_Z = Input((self.conf.num_z,))
        sample_X = self.Decoder([sample_S, sample_Z])

        z_model = Model(self.Enc_Modality.inputs, self.Enc_Modality.get_layer('z').output)
        rec_Z = z_model([sample_S, sample_X])
        self.Z_Regressor = Model(inputs=[sample_S, sample_Z], outputs=rec_Z, name='Z_Regressor')
        log.info('Z Regressor')
        self.Z_Regressor.summary(print_fn=log.info)

        self.Z_Regressor.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay), loss=['mae'],
                                 loss_weights=[self.w_rec_Z])

    def compile(self):
        self.G_trainer.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay), loss=['mse', 'mae', costs.ypred],
                               loss_weights=[self.conf.w_adv_M, self.w_rec_X, self.w_kl])
        self.G_supervised_trainer.compile(Adam(self.conf.lr, beta_1=0.5, decay=self.conf.decay),
                                          loss=[costs.make_dice_loss_fnc(self.loader.num_masks), 'mae', costs.ypred],
                                          loss_weights=[self.w_sup_M, self.w_rec_X, self.w_kl])

    def get_segmentor(self):
        inp = Input(self.conf.input_shape)
        return Model(inputs=inp, outputs=self.Segmentor(self.Enc_Anatomy(inp)))


