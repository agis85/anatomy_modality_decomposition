import itertools

import logging
import numpy as np
import os
from abc import abstractmethod
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

from callbacks.image_callback import SaveEpochImages
from costs import dice
from loaders import loader_factory
from utils.image_utils import save_segmentation

log = logging.getLogger('executor')


class Executor(object):
    """
    Base class for executor objects.
    """
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.loader = loader_factory.init_loader(self.conf.dataset_name)
        self.epoch = 0
        self.models_folder = self.conf.folder + '/models'
        self.train_data = None
        self.valid_data = None
        self.train_folder = None

    @abstractmethod
    def init_train_data(self):
        self.train_data = self.loader.load_labelled_data(self.conf.split, 'training',
                                                    downsample=self.conf.image_downsample, modality=self.conf.modality)
        self.valid_data = self.loader.load_labelled_data(self.conf.split, 'validation',
                                                    downsample=self.conf.image_downsample, modality=self.conf.modality)

        self.train_data.select_masks(self.conf.num_masks)
        self.valid_data.select_masks(self.conf.num_masks)

        self.train_data.sample(int(self.conf.l_mix * self.train_data.num_volumes), seed=self.conf.seed)
        self.conf.data_len = self.train_data.size()

    @abstractmethod
    def get_loss_names(self):
        pass

    @abstractmethod
    def train(self):
        log.info('Training Model')
        self.init_train_data()

        self.train_folder = os.path.join(self.conf.folder, 'training_results')
        if not os.path.exists(self.train_folder):
            os.mkdir(self.train_folder)

        callbacks = self.init_callbacks()

        train_images = self.get_inputs(self.train_data)
        train_labels = self.get_labels(self.train_data)

        valid_images = self.get_inputs(self.valid_data)
        valid_labels = self.get_labels(self.valid_data)

        if self.conf.outputs > 1:
            train_labels = [self.train_data.masks[..., i:i+1] for i in range(self.conf.outputs)]
            valid_labels = [self.valid_data.masks[..., i:i+1] for i in range(self.conf.outputs)]
        if self.conf.augment:
            datagen_dict = self.get_datagen_params()
            datagen = ImageDataGenerator(**datagen_dict)

            gen = data_generator_multiple_outputs(datagen, self.conf.batch_size, train_images, [train_labels])
            self.model.model.fit_generator(gen, steps_per_epoch=len(train_images) / self.conf.batch_size,
                                           epochs=self.conf.epochs, callbacks=callbacks,
                                           validation_data=(valid_images, valid_labels))
        else:
            self.model.model.fit(train_images, train_labels,
                                 validation_data=(valid_images, valid_labels),
                                 epochs=self.conf.epochs, callbacks=callbacks, batch_size=self.conf.batch_size)

    def init_callbacks(self):
        datagen_dict = self.get_datagen_params()
        img_gen = ImageDataGenerator(**datagen_dict).flow(self.train_data.images, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        msk_gen = ImageDataGenerator(**datagen_dict).flow(self.train_data.masks, batch_size=self.conf.batch_size,
                                                           seed=self.conf.seed)
        gen = itertools.zip_longest(img_gen, msk_gen)

        es = EarlyStopping(min_delta=0.001, patience=100)
        si = SaveEpochImages(self.conf, self.model, gen)
        cl = CSVLogger(self.train_folder + '/training.csv')
        mc = ModelCheckpoint(self.conf.folder + '/model', monitor='val_loss', verbose=0, save_best_only=False,
                             save_weights_only=True, mode='min', period=1)
        mc_best = ModelCheckpoint(self.conf.folder + '/model_best', monitor='val_loss', verbose=0, save_best_only=True,
                             save_weights_only=True, mode='min', period=1)
        return [es, si, cl, mc, mc_best]

    def get_labels(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's target, usually the masks
        """
        return data.masks

    def get_inputs(self, data):
        """
        :param data: the Data object used in training
        :return:     the network's input, usually the images
        """
        return data.images

    @abstractmethod
    def test(self):
        """
        Evaluate a model on the test data.
        """
        log.info('Evaluating model on test data')
        folder = os.path.join(self.conf.folder, 'test_results_%s' % self.conf.test_dataset)
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.test_modality(folder, self.conf.modality)

    def test_modality(self, folder, modality):
        test_loader = loader_factory.init_loader(self.conf.test_dataset)
        test_data = test_loader.load_labelled_data(self.conf.split, 'test', modality=modality,
                                                   downsample=self.conf.image_downsample)

        segmentor = self.model.get_segmentor()
        synth = []
        im_dice = {}
        samples = os.path.join(folder, 'samples')
        if not os.path.exists(samples):
            os.makedirs(samples)
        f = open(os.path.join(folder, 'results.csv'), 'w')
        f.writelines('Vol, Dice, ' + ', '.join(['Dice%d' % mi for mi in range(test_loader.num_masks)]) + '\n')
        for vol_i in test_data.volumes():
            vol_folder = os.path.join(samples, 'vol_%s' % str(vol_i))
            if not os.path.exists(vol_folder):
                os.makedirs(vol_folder)

            vol_image = test_data.get_images(vol_i)
            vol_mask = test_data.get_masks(vol_i)
            assert vol_image.shape[0] > 0 and vol_image.shape[:-1] == vol_mask.shape[:-1]
            pred = segmentor.predict(vol_image)

            synth.append(pred)
            im_dice[vol_i] = dice(vol_mask, pred)
            sep_dice = [dice(vol_mask[..., mi:mi + 1], pred[..., mi:mi + 1], binarise=True) for mi in
                        range(test_loader.num_masks)]

            s = '%s, %.3f, ' + ', '.join(['%.3f'] * test_loader.num_masks) + '\n'
            d = (str(vol_i), im_dice[vol_i]) + tuple(sep_dice)
            f.writelines(s % d)

            for i in range(vol_image.shape[0]):
                d, m = vol_image[i], vol_mask[i]
                save_segmentation(vol_folder, segmentor, d, m, name_prefix='test_vol%s_im%d' % (str(vol_i), i))
        print('Dice score: %.3f' % np.mean(list(im_dice.values())))
        f.close()

    def stop_criterion(self, es, logs):
        es.on_epoch_end(self.epoch, logs)
        if es.stopped_epoch > 0:
            return True

    def get_datagen_params(self):
        """
        Construct a dictionary of augmentations.
        :param augment_spatial:
        :param augment_intensity:
        :return: a dictionary of augmentation parameters to use with a keras image processor
        """
        result = dict(horizontal_flip=False, vertical_flip=False, rotation_range=0.)

        if self.conf.augment:
            result['rotation_range'] = 90.
            result['horizontal_flip'] = False
            result['vertical_flip'] = False
            result['width_shift_range'] = 0.0
            result['height_shift_range'] = 0.0

        return result

    def align_batches(self, array_list):
        """
        Align the arrays of the input list, based on batch size.
        :param array_list: list of 4-d arrays to align
        """
        mn = np.min([x.shape[0] for x in array_list])
        new_list = [x[0:mn] for x in array_list]
        return new_list

    def get_fake(self, pred, fake_pool, sample_size=-1):
        sample_size = self.conf.batch_size if sample_size == -1 else sample_size

        if pred.shape[0] > 0:
            fake_pool.extend(pred)

        fake_pool = fake_pool[-self.conf.pool_size:]
        sel = np.random.choice(len(fake_pool), size=(sample_size,), replace=False)
        fake_A = np.array([fake_pool[ind] for ind in sel])
        return fake_pool, fake_A


def data_generator_multiple_outputs(datagen, batch_size, inp, outs):
    gen_inp = datagen.flow(inp, seed=1, batch_size=batch_size)
    gen_outs = [datagen.flow(o, seed=1, batch_size=batch_size) for o in outs]
    while True:
        x = next(gen_inp)
        y = [next(gen_o) for gen_o in gen_outs]
        yield x, y
