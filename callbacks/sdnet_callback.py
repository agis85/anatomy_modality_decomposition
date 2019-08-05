
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from scipy.misc import imsave

log = logging.getLogger('SDNetCallback')


class SDNetCallback(Callback):
    """
    Image callback for saving images during SDNet training.
    Images are saved in a subfolder with name training_images, created inside the experiment folder.
    """
    def __init__(self, folder, batch_size, sdnet):
        """
        :param folder:      experiment folder, where all results are saved
        :param batch_size:  batch size used for training
        """
        super(SDNetCallback, self).__init__()

        # Create results folder
        self.folder = os.path.join(folder, 'training_images')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.sdnet      = sdnet
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, data_labelled=None, images_unlabelled=None, logs=None):
        """
        Overwrite default on_epoch_end implementation.

        :param epoch:           current training epoch
        :param data_labelled:   a list of tuples (image, mask)
        :param data_unlabelled: a list of images with no corresponding masks
        :param logs:            a dictionary of losses. Not used here
        """
        images_labelled   = np.concatenate([data_labelled[i][0] for i in range(len(data_labelled))], axis=0)
        masks             = np.concatenate([data_labelled[i][1] for i in range(len(data_labelled))], axis=0)
        images_unlabelled = np.array(images_unlabelled)

        self.plot_images(epoch, images_labelled, masks, images_unlabelled)
        self.plot_discriminator_outputs(epoch, np.concatenate([images_labelled, images_unlabelled]), masks)

    def plot_images(self, epoch, images_labelled, masks, images_unlabelled):
        """
        Save segmentation and reconstruction examples.
        :param epoch:           current training epoch
        :param images_labelled:   an array of labelled images
        :param masks:             an array of corresponding masks (to the labelled images)
        :param images_unlabelled: an array of images with no masks
        """
        rows = []
        # plot 3 labelled examples
        for i in range(3):
            rows.append(self.get_image_row(images_labelled, masks))
        # plot 3 unlabelled examples
        for i in range(3):
            rows.append(self.get_image_row(images_unlabelled, np.zeros(images_unlabelled.shape[:-1] + (2,))))

        img = np.concatenate(rows, axis=0)
        imsave(self.folder + '/cardiacgan_epoch_%d.png' % epoch, img)

    def get_image_row(self, images, masks):
        """
        Create an array of 8 images showing segmentations and reconstructions with different combinations of masks
        and residuals
        :param images: an array of images
        :param masks:  an array of masks
        :return:       a concatenated array of 8 subarrays to be used as one row of the final plotted image
        """
        if len(images) == 0:
            return []

        xi = np.random.randint(images.shape[0]) # draw random sample
        x = images[xi:xi + 1]
        pred_m, z = self.sdnet.Decomposer.predict(x)

        m = masks[xi:xi + 1]
        rec_predM_z = self.sdnet.Reconstructor.predict([pred_m, z])
        rec_m_z     = self.sdnet.Reconstructor.predict([m, z])
        rec_m0_z    = self.sdnet.Reconstructor.predict([np.zeros(m.shape), z])
        rec_m_z0    = self.sdnet.Reconstructor.predict([m, np.zeros(z.shape)])
        rec_m0_z0   = self.sdnet.Reconstructor.predict([np.zeros(m.shape), np.zeros(z.shape)])

        return np.concatenate([np.squeeze(el) for el in
                               [x, self.format_mask(pred_m), rec_predM_z, self.format_mask(m),
                                rec_m_z, rec_m0_z, rec_m_z0, rec_m0_z0]], axis=1)

    def format_mask(self, m):
        result = np.zeros(shape=(m.shape[1], m.shape[2]))
        for j in range(m.shape[-1]):
            result += m[0, :, :, j] * (0.2 * (j + 1))
        return result

    def plot_discriminator_outputs(self, epoch, images, masks):
        if masks.shape[0] == 0:
            return

        # number of points used for the histogram
        sz = 40 if 40 < np.min([len(images), len(masks)]) else np.min([len(images), len(masks)])
        idx_X = np.random.choice(len(images), size=sz, replace=False)
        idx_M = np.random.choice(len(masks), size=sz, replace=False)

        samples_X = np.concatenate([images[i:i + 1] for i in idx_X], axis=0)
        samples_M = np.concatenate([masks[i:i + 1] for i in idx_M], axis=0)
        samples_pred_M, samples_Z = self.sdnet.Decomposer.predict(samples_X)
        samples_pred_X = self.sdnet.Reconstructor.predict([samples_pred_M, samples_Z])

        dx_true = np.array([np.mean(self.sdnet.ImageDiscriminator.predict(samples_X[i:i + 1])) for i in range(sz)])
        dx_pred = np.array([np.mean(self.sdnet.ImageDiscriminator.predict(samples_pred_X[i:i + 1])) for i in range(sz)])
        dm_true = np.array([np.mean(self.sdnet.MaskDiscriminator.predict(samples_M[i:i + 1])) for i in range(sz)])
        dm_pred = np.array([np.mean(self.sdnet.MaskDiscriminator.predict(samples_pred_M[i:i + 1])) for i in range(sz)])

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.hist([dx_true, dx_pred], stacked=True, normed=True)
        plt.subplot(1, 2, 2)
        plt.hist([dm_true, dm_pred], stacked=True, normed=True)
        plt.savefig(self.folder + '/discriminator_hist_epoch_%d.png' % epoch)
        plt.close()
