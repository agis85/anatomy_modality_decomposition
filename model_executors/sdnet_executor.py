import itertools
import logging
import os

import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Progbar

import costs
import utils.data_utils
from callbacks.loss_callback import SaveLoss
from callbacks.sdnet_image_callback import SDNetImageCallback
from model_executors.base_executor import Executor
from utils.distributions import NormalDistribution

log = logging.getLogger('sdnet_executor')


class SDNetExecutor(Executor):
    """
    Executor for training SDNet.
    """
    def __init__(self, conf, model):
        super(SDNetExecutor, self).__init__(conf, model)
        self.model = model

        self.S_pool = []  # Pool of anatomy maps
        self.X_pool = []  # Pool of images
        self.M_pool = []  # Pool of masks
        self.Z_pool = []  # Pool of latent vectors

        self.gen_unlabelled = None
        self.discriminator_masks = None
        self.img_clb = None
        self.data = None

    def test(self):
        """
        Evaluate a model on the test data.
        """
        if self.conf.modality == 'all':
            for modality in self.loader.modalities:
                log.info('Evaluating model on test data %s' % modality)
                folder = os.path.join(self.conf.folder, 'test_results_%s_%s' % (self.conf.test_dataset, modality))
                if not os.path.exists(folder):
                    os.makedirs(folder)

                self.test_modality(folder, modality)
        else:
            super(SDNetExecutor, self).test()

    def init_train_data(self):
        """
        Initialise data iterators.
        :param split_type: training/validation/test
        """
        self.gen_labelled         = self._init_labelled_data_generator()
        self.gen_unlabelled       = self._init_unlabelled_data_generator()
        self.discriminator_masks  = self._init_disciminator_mask_generator()
        self.discriminator_images = self._init_discriminator_image_generator()

        self.conf.batches = int(np.ceil(self.conf.data_len / self.conf.batch_size))

    def _init_labelled_data_generator(self):
        """
        Initialise a data generator (image, mask, scanner) for labelled data
        """
        if self.conf.l_mix == 0:
            return

        log.info('Initialising labelled datagen. Loading %s data' % self.conf.dataset_name)
        self.data = self.loader.load_labelled_data(self.conf.split, 'training', modality=self.conf.modality,
                                                   downsample=self.conf.image_downsample)
        if self.conf.modality == 'all':
            data1 = self.data.copy()
            data1.filter_by_scanner(self.loader.modalities[0]) # mr
            data2 = self.data.copy()
            data2.filter_by_scanner(self.loader.modalities[1]) # mr
            assert data1.size() == data2.size(), 'Unequal sizes: %d vs %d' % (data1.size(), data2.size())

            data1.sample(int(self.conf.l_mix * self.data.num_volumes), seed=self.conf.seed)
            data2.sample(int(self.conf.l_mix2 * self.data.num_volumes), seed=self.conf.seed)
            data1.merge(data2)
            self.data = data1.copy()
        else:
            self.data.sample(int(self.conf.l_mix * self.data.num_volumes), seed=self.conf.seed)
        self.data.crop(self.conf.input_shape[:2]) # crop data to input shape: useful in transfer learning
        self.conf.data_len = self.data.size()

        datagen_dict1 = self.get_datagen_params()
        datagen_dict2 = self.get_datagen_params()
        img_gen = ImageDataGenerator(**datagen_dict1).flow(self.data.images, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        msk_gen = ImageDataGenerator(**datagen_dict2).flow(self.data.masks, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        scn_gen = utils.data_utils.generator(self.conf.batch_size, self.conf.seed, 'no_overflow', self.data.scanner)
        return itertools.zip_longest(img_gen, msk_gen, scn_gen)

    def _init_unlabelled_data_generator(self):
        """
        Initialise a data generator (image) for unlabelled data
        """
        if self.conf.ul_mix == 0:
            return

        ul_data = self._load_unlabelled_data('ul')

        self.conf.unlabelled_image_num = ul_data.size()
        if self.data is None or ul_data.size() > self.data.size():
            self.conf.data_len = ul_data.size()

        datagen_dict = self.get_datagen_params()
        return ImageDataGenerator(**datagen_dict) \
            .flow(ul_data.images, batch_size=self.conf.batch_size, seed=self.conf.seed)

    def _load_unlabelled_data(self, data_type):
        '''
        Create a Data object with unlabelled data. This will be used to train the unlabelled path of the
        generators and produce fake masks for training the discriminator
        :param data_type:   can be one ['ul', 'all']. The second includes images that have masks.
        :return:            a data object
        '''
        log.info('Loading unlabelled images of type %s' % data_type)
        log.info('Estimating number of unlabelled images from %s data' % self.conf.dataset_name)

        num_labelled_volumes = len(self.loader.splits()[self.conf.split]['training'])
        ul_mix = 1 if self.conf.ul_mix > 1 else self.conf.ul_mix
        self.conf.num_ul_volumes = int(num_labelled_volumes * ul_mix)
        log.info('Sampling %d unlabelled images out of total %d.' % (self.conf.num_ul_volumes, num_labelled_volumes))

        log.info('Initialising unlabelled datagen. Loading %s data' % self.conf.dataset_name)
        if data_type == 'ul':
            ul_data = self.loader.load_unlabelled_data(self.conf.split, 'training', modality=self.conf.modality)
            ul_data.crop(self.conf.input_shape[:2])
        elif data_type == 'all':
            ul_data = self.loader.load_all_data(self.conf.split, 'training', modality=self.conf.modality)
            ul_data.crop(self.conf.input_shape[:2])
        else:
            raise Exception('Invalid data_type: %s' % str(data_type))
        ul_data.sample(self.conf.num_ul_volumes, seed=self.conf.seed)

        # Use 1200 unlabelled images maximum, to be comparable with the total number of labelled images of ACDC (~1200)
        max_ul_images = 1200 if self.conf.ul_mix <= 1 else 1200 * self.conf.ul_mix
        if ul_data.size() > max_ul_images:
            samples_per_volume = int(np.ceil(max_ul_images / ul_data.num_volumes))
            ul_data.sample_per_volume(samples_per_volume, seed=self.conf.seed)
        return ul_data

    def _init_disciminator_mask_generator(self, batch_size=None):
        """
        Init a generator for masks to use in the discriminator.
        """
        log.info('Initialising discriminator maskgen.')
        masks = self._load_discriminator_masks()

        datagen_dict = self.get_datagen_params()
        other_datagen = ImageDataGenerator(**datagen_dict)
        bs = self.conf.batch_size if batch_size is None else batch_size
        return other_datagen.flow(masks, batch_size=bs, seed=self.conf.seed)

    def _load_discriminator_masks(self):
        """
        :return: dataset masks
        """
        temp = self.loader.load_labelled_data(self.conf.split, 'training', modality=self.conf.modality,
                                              downsample=self.conf.image_downsample)
        temp.crop(self.conf.input_shape[:2])
        masks = temp.masks.copy()
        del temp

        im_shape = self.conf.input_shape[:2]
        assert masks.shape[1] == im_shape[0] and masks.shape[2] == im_shape[1], masks.shape
        return masks

    def _init_discriminator_image_generator(self):
        """
        Init a generator for images to train a discriminator (for fake masks)
        """
        if self.conf.ul_mix == 0:
            return

        log.info('Initialising discriminator imagegen.')
        data = self._load_unlabelled_data('all')
        datagen_dict = self.get_datagen_params()
        datagen = ImageDataGenerator(**datagen_dict)
        return datagen.flow(data.images, batch_size=self.conf.batch_size, seed=self.conf.seed)

    def init_image_callback(self):
        log.info('Initialising a data generator to use for printing.')
        datagen_dict1 = self.get_datagen_params()
        datagen_dict2 = self.get_datagen_params()
        if self.gen_labelled is None:
            assert self.data is None
            self.data = self.loader.load_labelled_data(self.conf.split, 'training', self.conf.modality,
                                                       downsample=self.conf.image_downsample)
            self.data.crop(self.conf.input_shape[:2])  # crop data to input shape: useful in transfer learning
            
        gen = itertools.zip_longest(
            ImageDataGenerator(**datagen_dict1).flow(self.data.images, batch_size=30, seed=self.conf.seed),
            ImageDataGenerator(**datagen_dict2).flow(self.data.masks, batch_size=30, seed=self.conf.seed))
        gen_ul = None if self.conf.ul_mix == 0 else self._init_unlabelled_data_generator()
        other_masks_gen = self._init_disciminator_mask_generator(batch_size=30)
        self.img_clb = SDNetImageCallback(self.conf, self.model, gen, gen_ul, other_masks_gen)

    def get_loss_names(self):
        """
        :return: loss names to report.
        """
        return ['adv_M', 'rec_X', 'dis_M', 'val_loss', 'supervised_Mask', 'loss', 'KL', 'rec_Z']

    def train(self):
        log.info('Training Model')

        self.init_train_data()

        self.init_image_callback()
        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder + '/training.csv')
        cl.on_train_begin()

        es = EarlyStopping('val_loss', min_delta=0.01, patience=100)
        es.model = self.model.Segmentor
        es.on_train_begin()

        loss_names = self.get_loss_names()
        total_loss = {n: [] for n in loss_names}

        progress_bar = Progbar(target=self.conf.batches * self.conf.batch_size)

        for self.epoch in range(self.conf.epochs):
            log.info('Epoch %d/%d' % (self.epoch, self.conf.epochs))

            epoch_loss = {n: [] for n in loss_names}
            epoch_loss_list = []

            D_initial_weights = np.mean([np.mean(w) for w in self.model.D_trainer.get_weights()])
            G_initial_weights = np.mean([np.mean(w) for w in self.model.G_trainer.get_weights()])
            for self.batch in range(self.conf.batches):
                # real_pools = self.add_to_pool(data, real_pools)
                self.train_batch(epoch_loss)

                progress_bar.update((self.batch + 1) * self.conf.batch_size)

            G_final_weights = np.mean([np.mean(w) for w in self.model.G_trainer.get_weights()])
            D_final_weights = np.mean([np.mean(w) for w in self.model.D_trainer.get_weights()])

            assert self.gen_unlabelled is None or not self.model.D_trainer.trainable \
                   or D_initial_weights != D_final_weights
            assert G_initial_weights != G_final_weights

            self.validate(epoch_loss)

            for n in loss_names:
                epoch_loss_list.append((n, np.mean(epoch_loss[n])))
                total_loss[n].append(np.mean(epoch_loss[n]))
            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.3f' for l in loss_names])) %
                     ((self.epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}

            cl.model = self.model.D_Mask
            cl.model.stop_training = False
            cl.on_epoch_end(self.epoch, logs)
            sl.on_epoch_end(self.epoch, logs)

            # Plot some example images
            self.img_clb.on_epoch_end(self.epoch)

            self.model.save_models()

            if self.stop_criterion(es, logs):
                log.info('Finished training from early stopping criterion')
                break

    def validate(self, epoch_loss):
        # Report validation error
        valid_data = self.loader.load_labelled_data(self.conf.split, 'validation', modality=self.conf.modality,
                                                    downsample=self.conf.image_downsample)
        valid_data.crop(self.conf.input_shape[:2])

        mask = self.model.Segmentor.predict(self.model.Enc_Anatomy.predict(valid_data.images))
        assert mask.shape[:-1] == valid_data.images.shape[:-1], str(valid_data.images.shape) + ' ' + str(mask.shape)
        epoch_loss['val_loss'].append((1 - costs.dice(valid_data.masks, mask)))

    def train_batch(self, epoch_loss):
        self.train_batch_generators(epoch_loss)
        self.train_batch_mask_discriminator(epoch_loss)

    def train_batch_generators(self, epoch_loss):
        """
        Train generator/segmentation networks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        if self.gen_labelled is not None:
            x, m, scanner = next(self.gen_labelled)
            batch_size = x.shape[0]  # maybe this differs from conf.batch_size at the last batch.

            # Train labelled path (G_supervised_model)
            h = self.model.G_supervised_trainer.fit(x, [m, x, np.zeros(batch_size)], epochs=1, verbose=0)
            epoch_loss['supervised_Mask'].append(h.history['Segmentor_loss'])
            epoch_loss['rec_X'].append(h.history['Reconstructor_loss'])
            epoch_loss['KL'].append(h.history['Enc_Modality_loss'])

            # Train Z Regressor
            if self.model.Z_Regressor is not None:
                s = self.model.Enc_Anatomy.predict(x)
                sample_z = NormalDistribution().sample((batch_size, self.conf.num_z))
                h = self.model.Z_Regressor.fit([s, sample_z], sample_z, epochs=1, verbose=0)
                epoch_loss['rec_Z'].append(h.history['loss'])

        # Train unlabelled path
        if self.gen_unlabelled is not None:
            x = next(self.gen_unlabelled)
            batch_size = x.shape[0]  # maybe this differs from conf.batch_size at the last batch.

            # Train unlabelled path (G_model)
            h = self.model.G_trainer.fit(x, [np.ones((batch_size,) + self.model.D_Mask.output_shape[1:]),
                                             x, np.zeros(batch_size)], epochs=1, verbose=0)
            epoch_loss['adv_M'].append(h.history['D_Mask_loss'])
            epoch_loss['rec_X'].append(h.history['Reconstructor_loss'])
            epoch_loss['KL'].append(h.history['Enc_Modality_loss'])

            # Train Z Regressor
            if self.model.Z_Regressor is not None:
                s = self.model.Enc_Anatomy.predict(x)
                sample_z = NormalDistribution().sample((batch_size, self.conf.num_z))
                h = self.model.Z_Regressor.fit([s, sample_z], sample_z, epochs=1, verbose=0)
                epoch_loss['rec_Z'].append(h.history['loss'])

    def train_batch_mask_discriminator(self, epoch_loss):
        """
        Jointly train a discriminator for masks.
        :param epoch_loss:  Dictionary of losses for the epoch
        """
        m = next(self.discriminator_masks)
        x = next(self.discriminator_images)
        x, m = self.align_batches([x, m])
        batch_size = m.shape[0]  # maybe this differs from conf.batch_size at the last batch.

        fake_s = self.model.Enc_Anatomy.predict(x)
        fake_m = self.model.Segmentor.predict(fake_s)

        # Pool of fake images
        self.M_pool, fake_m = self.get_fake(fake_m, self.M_pool, sample_size=batch_size)

        # Train Discriminator
        m_shape = (batch_size,) + self.model.D_Mask.get_output_shape_at(0)[1:]
        h = self.model.D_trainer.fit([m, fake_m], [np.ones(m_shape), np.zeros(m_shape)], epochs=1, verbose=0)
        epoch_loss['dis_M'].append(np.mean(h.history['loss']))
