import logging
import os
from abc import abstractmethod

import numpy as np
from keras.callbacks import Callback

from costs import dice
from utils.image_utils import save_segmentation, save_multiimage_segmentation

log = logging.getLogger('BaseSaveImage')


class BaseSaveImage(Callback):
    """
    Abstract base class for saving training images
    """
    def __init__(self, folder, model):
        super(BaseSaveImage, self).__init__()
        self.folder = os.path.join(folder, 'training_images')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.model = model

    @abstractmethod
    def on_epoch_end(self, epoch=None, logs=None):
        pass


class SaveImage(Callback):
    """
    Simple callback that saves segmentation masks and dice error.
    """
    def __init__(self, folder, test_data, test_masks=None, input_len=None):
        super(SaveImage, self).__init__()
        self.folder = folder
        self.test_data = test_data  # this can be a list of images of different spatial dimensions
        self.test_masks = test_masks
        self.input_len = input_len

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        all_dice = []
        for i in range(len(self.test_data)):
            d, m = self.test_data[i], self.test_masks[i]
            s = save_segmentation(self.folder, self.model, d, m, 'slc_%d' % i)
            all_dice.append(-dice(self.test_masks[i:i+1], s))

        f = open(os.path.join(self.folder, 'test_error.txt'), 'a+')
        f.writelines("%d, %.3f\n" % (epoch, np.mean(all_dice)))
        f.close()


class SaveEpochImages(Callback):
    def __init__(self, conf, model, gen):
        super(SaveEpochImages, self).__init__()
        self.folder = conf.folder + '/training'
        self.conf = conf
        self.model = model
        self.gen = gen
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def on_epoch_end(self, epoch, logs=None):
        x, m = next(self.gen)
        y = self.model.predict(x)
        save_multiimage_segmentation(x, m, y, self.folder, epoch)

