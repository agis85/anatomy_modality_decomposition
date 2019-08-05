import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from keras import Input, Model
from scipy.misc import imsave

import utils.data_utils
import utils.image_utils
from callbacks.image_callback import BaseSaveImage
from layers.rounding import Rounding
from utils import sdnet_utils
from utils.distributions import NormalDistribution
from utils.sdnet_utils import get_net

log = logging.getLogger('SDNetImageCallback')


class SDNetImageCallback(BaseSaveImage):
    def __init__(self, conf, sdnet, data_gen_lb, data_gen_ul, mask_gen):
        '''
        Callback for printint various types of images during SDNet training.

        :param folder:         location of callback images
        :param generators:     a list of "generator-tye" NN: usually [Decomposer, Reconstructor, Segmentor]
        :param discriminators: a list of discriminator NN: usually  [D_Images, D_Masks, D_Z]
        :param data_gen_lb:    a python iterator of images+masks
        :param data_gen_ul:    a python iterator of images
        :param mask_gen:       a python iterator of additional masks with full anatomy used in discriminator: can be None
        '''
        self.conf = conf
        super(SDNetImageCallback, self).__init__(conf.folder, sdnet)

        self._make_dirs(self.folder)
        self.data_gen_lb = data_gen_lb
        self.data_gen_ul = data_gen_ul
        self.mask_gen = mask_gen
        self.init_models()

    def _make_dirs(self, folder):
        self.lr_folder = folder + '/images_lr'
        if not os.path.exists(self.lr_folder):
            os.makedirs(self.lr_folder)

        self.segm_folder = folder + '/images_segm'
        if not os.path.exists(self.segm_folder):
            os.makedirs(self.segm_folder)

        self.rec_folder = folder + '/images_rec'
        if not os.path.exists(self.rec_folder):
            os.makedirs(self.rec_folder)

        self.discr_folder = folder + '/images_discr'
        if not os.path.exists(self.discr_folder):
            os.makedirs(self.discr_folder)

        self.interp_folder = folder + '/images_interp'
        if not os.path.exists(self.interp_folder):
            os.makedirs(self.interp_folder)

    def init_models(self):
        self.enc_anatomy = self.model.Enc_Anatomy
        self.reconstructor = self.model.Decoder
        self.segmentor = self.model.Segmentor
        self.discr_mask = self.model.D_Mask
        self.enc_modality = self.model.Enc_Modality

        mean = get_net(self.enc_modality, 'z_mean')
        var = get_net(self.enc_modality, 'z_log_var')
        self.z_mean = Model(self.enc_modality.inputs, mean.output)
        self.z_var = Model(self.enc_modality.inputs, var.output)

        inp = Input(self.conf.input_shape)
        self.round_model = Model(inp, Rounding()(self.enc_anatomy(inp)))

    def on_epoch_end(self, epoch=None, logs=None):
        '''
        Plot training images from the real_pool. For SDNet the real_pools will contain images paired with masks,
        and also unlabelled images.
        :param epoch:       current training epoch
        :param real_pools:  pool of images. Each element might be an image or a real mask
        :param logs:
        '''
        lb_images = next(self.data_gen_lb)

        # we usually plot 4 image-rows. If we have less, it means we've reached the end of the data, so iterate from
        # the beginning
        if len(lb_images[0]) < 4:
            lb_images = next(self.data_gen_lb)

        ul_images = []
        if self.data_gen_ul is not None:
            ul_images = next(self.data_gen_ul)
            _, b = utils.data_utils.crop_same([ul_images], [ul_images],
                                              size=(lb_images[0].shape[1], lb_images[0].shape[2]))
            ul_images = b[0]

        masks = None if self.mask_gen is None else next(self.mask_gen)
        if masks is not None:
            if len(masks) < 4:
                masks = next(self.mask_gen)
            _, b = utils.data_utils.crop_same([masks], [masks], size=(lb_images[0].shape[1], lb_images[0].shape[2]))
            masks = b[0]

        self.plot_latent_representation(lb_images, ul_images, epoch)
        self.plot_segmentations(lb_images, ul_images, epoch)
        self.plot_reconstructions(lb_images, ul_images, epoch)
        self.plot_discriminator_outputs(lb_images, ul_images, masks, epoch)
        self.plot_image_switch_lr(lb_images, ul_images, epoch)
        self.plot_image_interpolation(lb_images, ul_images, epoch)
        # self.plot_image_augmentations(lb_images, ul_images, epoch)

    # def _process_image_pool(self, real_pool):
    #     lb_images = []
    #     ul_images = []
    #
    #     i = 0
    #     while i < len(real_pool):
    #         cur_el = real_pool[i]
    #
    #         # we are reading the last element of the pool, so this must be an unlabelled image,
    #         # otherwise we wouldn't get to this point.
    #         if i + 1 == len(real_pool):
    #             assert cur_el.shape[-1] == 1
    #             ul_images.append(cur_el)
    #             return lb_images, ul_images
    #
    #         nxt_el = real_pool[i+1]
    #
    #         # if we find image + mask, append both to the labelled list
    #         if cur_el.shape[-1] == 1 and nxt_el.shape[-1] > 1:
    #             lb_images.append([cur_el, nxt_el])
    #             i += 2
    #         # if we find image + image, append the first to the unlabelled list
    #         elif cur_el.shape[-1] == 1 and nxt_el.shape[-1] == 1:
    #             ul_images.append(cur_el)
    #             i += 1
    #
    #     return lb_images, ul_images


    def plot_latent_representation(self, lb_images, ul_images, epoch):
        """
        Plot a 4-row image, where the first column shows the input image and the following columns
        each of the 8 channels of the spatial latent representation.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch    :   the epoch number
        """

        # combine labelled and unlabelled images and randomly sample 4 examples
        images = lb_images[0]  # [el[0] for el in lb_images]
        if len(ul_images) > 0:
            images = np.concatenate([images, ul_images], axis=0)
            x = np.concatenate([images[0:2], ul_images[0:2]], axis=0)
        else:
            x = utils.data_utils.sample(images, nb_samples=4, seed=self.conf.seed)

        # plot S
        s = self.enc_anatomy.predict(x)

        rows = [np.concatenate([x[i, :, :, 0]] + [s[i, :, :, s_chn] for s_chn in range(s.shape[-1])], axis=1)
                for i in range(x.shape[0])]
        im_plot = np.concatenate(rows, axis=0)
        scipy.misc.imsave(self.lr_folder + '/s_lr_epoch_%d.png' % epoch, im_plot)

        plt.figure()
        plt.imshow(im_plot, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.close()

        if self.conf.rounding == 'decoder':
            s = self.round_model.predict(x)
            rows = [np.concatenate([x[i, :, :, 0]] + [s[i, :, :, s_chn] for s_chn in range(s.shape[-1])], axis=1)
                   for i in range(x.shape[0])]
            im_plot = np.concatenate(rows, axis=0)
            scipy.misc.imsave(self.lr_folder + '/srnd_lr_epoch_%d.png' % epoch, im_plot)

            plt.figure()
            plt.imshow(im_plot, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.close()

        # plot Z
        enc_modality_inputs = [self.enc_anatomy.predict(images), images]
        z, _ = self.enc_modality.predict(enc_modality_inputs)
        gaussian = NormalDistribution()
        real_z = gaussian.sample(z.shape)

        fig, axes = plt.subplots(nrows=z.shape[1], ncols=2, sharex=True, sharey=True, figsize=(10, 8))
        axes[0, 0].set_title('Predicted Z')
        axes[0, 1].set_title('Real Z')
        for i in range(len(axes)):
            axes[i, 0].hist(z[:, i], normed=True, bins=11, range=(-3, 3))
            axes[i, 1].hist(real_z[:, i], normed=True, bins=11, range=(-3, 3))
        axes[0, 0].plot(0, 0)

        plt.savefig(self.lr_folder + '/z_lr_epoch_%d.png' % epoch)
        plt.close()

        means = self.z_mean.predict(enc_modality_inputs)
        variances  = self.z_var.predict(enc_modality_inputs)
        means = np.var(means, axis=0)
        variances = np.mean(np.exp(variances), axis=0)
        with open(self.lr_folder + '/z_means.csv', 'a+') as f:
            f.writelines(', '.join([str(means[i]) for i in range(means.shape[0])]) + '\n')
        with open(self.lr_folder + '/z_vars.csv', 'a+') as f:
            f.writelines(', '.join([str(variances[i]) for i in range(variances.shape[0])]) + '\n')

    def plot_segmentations(self, lb_images, ul_images, epoch):
        '''
        Plot an image for every sample, where every row contains a channel of the spatial LR and a channel of the
        predicted mask.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''

        imags = lb_images[0]  # [el[0] for el in lb_images]
        masks = lb_images[1]  # [el[1] for el in lb_images]

        x = utils.data_utils.sample(imags, 4, seed=self.conf.seed)
        m = utils.data_utils.sample(masks, 4, seed=self.conf.seed)

        if len(ul_images) > 0:
            x_ul = utils.data_utils.sample(imags, 4, seed=self.conf.seed)
            m_ul = np.array([np.zeros(shape=m[0].shape) for i in range(4)])
            x = np.concatenate([x, x_ul], axis=0)
            m = np.concatenate([m, m_ul], axis=0)

        assert x.shape[:-1] == m.shape[:-1], 'Incompatible shapes: %s vs %s' % (str(x.shape), str(m.shape))

        s = self.enc_anatomy.predict(x)
        y = self.segmentor.predict(s)

        rows = []
        for i in range(x.shape[0]):
            y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
            m_list = [m[i, :, :, chn] for chn in range(m.shape[-1])]
            if m.shape[-1] < y.shape[-1]:
                m_list += [np.zeros(shape=(m.shape[1], m.shape[2]))] * (y.shape[-1] - m.shape[-1])
            assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))
            rows += [np.concatenate([x[i, :, :, 0]] + y_list + m_list, axis=1)]

        im_plot = np.concatenate(rows, axis=0)
        scipy.misc.imsave(self.segm_folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)

    def plot_reconstructions(self, lb_images, ul_images, epoch):
        """
        Plot two images showing the combination of the spatial and modality LR to generate an image. The first
        image uses the predicted S and Z and the second samples Z from a Gaussian.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        """

        # combine labelled and unlabelled images and randomly sample 4 examples
        images = lb_images[0]  # [el[0] for el in lb_images]
        if len(ul_images) > 0:
            images = np.concatenate([images, ul_images], axis=0)
        x = utils.data_utils.sample(images, nb_samples=4)

        # S + Z -> Image
        gaussian = NormalDistribution()

        s = self.enc_anatomy.predict(x)
        z, _ = self.enc_modality.predict([s, x])

        y = self.reconstructor.predict([s, z])
        y_s0 = self.reconstructor.predict([s, np.zeros(z.shape)])
        all_bkg = np.concatenate([np.zeros(s.shape[:-1] + (s.shape[-1] - 1,)), np.ones(s.shape[:-1] + (1,))], axis=-1)
        y_0z = self.reconstructor.predict([all_bkg, z])
        y_00 = self.reconstructor.predict([all_bkg, np.zeros(z.shape)])
        z_random = gaussian.sample(z.shape)
        y_random = self.reconstructor.predict([s, z_random])

        rows = [np.concatenate([x[i, :, :, 0], y[i, :, :, 0], y_random[i, :, :, 0], y_s0[i, :, :, 0]] +
                               [self.reconstructor.predict([self._get_s0chn(k, s), z])[i, :, :, 0] for k in
                                range(s.shape[-1] - 1)] +
                               [y_0z[i, :, :, 0], y_00[i, :, :, 0]], axis=1) for i in range(x.shape[0])]
        header = utils.image_utils.makeTextHeaderImage(x.shape[2], ['X', 'rec(s,z)', 'rec(s,~z)', 'rec(s,0)'] +
                                                       ['rec(s0_%d, z)' % k for k in range(s.shape[-1] - 1)] + [
                                                        'rec(0, z)', 'rec(0,0)'])
        im_plot = np.concatenate([header] + rows, axis=0)
        im_plot = np.clip(im_plot, -1, 1)
        scipy.misc.imsave(self.rec_folder + '/rec_epoch_%d.png' % epoch, im_plot)

        plt.figure()
        plt.imshow(im_plot, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.close()

    def _get_s0chn(self, k, s):
        s_res = s.copy()
        chnk = s_res[..., k]
        # move channel k 1s to the background
        s_res[..., -1][chnk == 1] = 1
        s_res[..., k] = 0
        return s_res

    def plot_discriminator_outputs(self, lb_images, ul_images, other_masks, epoch):
        '''
        Plot a histogram of predicted values by the discriminator
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param other_masks: a 4-dim array of masks with full anatomy: can be None
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        masks = lb_images[1]  # [el[1] for el in lb_images]

        # when other_masks is provided, use it for GT discriminator results
        if other_masks is not None:
            masks = other_masks

        if len(ul_images) > 0:
            imags = np.concatenate([imags, ul_images], axis=0)

        # x = util_functions.sample(imags, 4, seed=0)
        # m = util_functions.sample(masks, 4, seed=0)
        x = imags
        m = masks
        print(m.shape)
        print(self.discr_mask.input_shape[-1], self.model.Segmentor.output_shape[-1])
        # Transfer learning case, where we fine-tune a model having masks of less segmentations
        # than the one the original model was trained with.
        if self.discr_mask.input_shape[-1] > m.shape[-1]:
            m = np.concatenate(
                [m, np.zeros(shape=m.shape[:-1] + ((self.model.Segmentor.output_shape[-1] - m.shape[-1]),))], axis=-1)
        elif self.discr_mask.input_shape[-1] < m.shape[-1]:
            m = m[..., 0:self.discr_mask.input_shape[-1]]
        print(m.shape)

        s = self.enc_anatomy.predict(x)
        pred_z, _ = self.enc_modality.predict([s, x])
        pred_m = self.segmentor.predict(s)
        # y = self.reconstructor.predict([s, pred_z])
        # Transfer learning case, where we fine-tune a model having masks of less segmentations
        # than the one the original model was trained with.
        if self.discr_mask.input_shape[-1] > pred_m.shape[-1]:
            pred_m = np.concatenate(
                [pred_m, np.zeros(shape=m.shape[:-1] + ((self.model.Segmentor.output_shape[-1] - pred_m.shape[-1]),))], axis=-1)
        elif self.discr_mask.input_shape[-1] < pred_m.shape[-1]:
            pred_m = pred_m[..., 0:self.discr_mask.input_shape[-1]]

        dm_input_fake = pred_m
        dm_true = self.discr_mask.predict(m).reshape(m.shape[0], -1).mean(axis=1)
        dm_pred = self.discr_mask.predict(dm_input_fake).reshape(pred_m.shape[0], -1).mean(axis=1)

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.title('Mask Discriminator')
        plt.hist([dm_true, dm_pred], stacked=True, normed=True)
        plt.savefig(self.discr_folder + '/discriminator_hist_epoch_%d.png' % epoch)
        plt.close()

        plt.figure()
        for i in range(4):
            plt.subplot(4, 2, 2 * i + 1)
            m_allchn = np.concatenate([m[i, :, :, chn] for chn in range(m.shape[-1])], axis=1)
            plt.imshow(m_allchn, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_mask.predict(m[i:i + 1]).reshape(1, -1).mean(axis=1))

            plt.subplot(4, 2, 2 * i + 2)
            pred_m_allchn = pred_m[i:i + 1]
            pred_m_allchn_img = np.concatenate([pred_m_allchn[0, :, :, chn] for chn in range(pred_m_allchn.shape[-1])],
                                               axis=1)
            plt.imshow(pred_m_allchn_img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('Pred: %.3f' % self.discr_mask.predict(pred_m_allchn).reshape(1, -1).mean(axis=1))
        plt.tight_layout()
        plt.savefig(self.discr_folder + '/discriminator_mask_epoch_%d.png' % epoch)
        plt.close()

    def plot_image_switch_lr(self, lb_images, ul_images, epoch):
        '''
        Switch anatomy between two images and plot the synthetic result
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        if len(ul_images) > 0:
            imags = np.concatenate([imags, ul_images], axis=0)

        x = utils.data_utils.sample(imags, 4, seed=self.conf.seed)

        rows = []
        for i in range(0, 4, 2):
            x1 = x[i: i + 1]
            x2 = x[i + 1: i + 2]

            s1 = self.enc_anatomy.predict(x1)
            z1, _ = self.enc_modality.predict([s1, x1])
            s2 = self.enc_anatomy.predict(x2)
            z2, _ = self.enc_modality.predict([s2, x2])

            x11 = self.reconstructor.predict([s1, z1])
            x12 = self.reconstructor.predict([s1, z2])
            x21 = self.reconstructor.predict([s2, z1])
            x22 = self.reconstructor.predict([s2, z2])

            row = np.concatenate([x1[0, :, :, 0], x11[0, :, :, 0], x12[0, :, :, 0], x21[0, :, :, 0], x22[0, :, :, 0],
                                  x2[0, :, :, 0]], axis=1)
            rows.append(row)

        header = utils.image_utils.makeTextHeaderImage(x.shape[2],
                                                       ['X1', 'Rec(s1,z1)', 'Rec(s1,z2)', 'Rec(s2,z1)', 'Rec(s2,z2)',
                                                     'X2'])
        image = np.concatenate([header] + rows, axis=0)
        imsave(self.interp_folder + '/switch_lr_epoch_%d.png' % (epoch), image)

    def plot_image_interpolation(self, lb_images, ul_images, epoch):
        '''
        Interpolate between two images and plot the transition in reconstructing the image.
        :param lb_images:   a list of 2 4-dim arrays of images + corresponding masks
        :param ul_images:   a list of 4-dim image arrays
        :param epoch:       the epoch number
        '''
        imags = lb_images[0]  # [el[0] for el in lb_images]
        if len(ul_images) > 0:
            imags = np.concatenate([imags, ul_images], axis=0)

        x = utils.data_utils.sample(imags, 4, seed=self.conf.seed)

        for i in range(0, 4, 2):
            x1 = x[i: i + 1]
            s1 = self.enc_anatomy.predict(x1)
            z1 = sdnet_utils.vae_sample([self.z_mean.predict([s1, x1]), self.z_var.predict([s1, x1])])

            x2 = x[i + 1: i + 2]
            s2 = self.enc_anatomy.predict(x2)
            z2 = sdnet_utils.vae_sample([self.z_mean.predict([s2, x2]), self.z_var.predict([s2, x2])])

            imsave(self.interp_folder + '/interpolation1_epoch_%d.png' % epoch, self._interpolate(s1, z1, z2))
            imsave(self.interp_folder + '/interpolation2_epoch_%d.png' % epoch, self._interpolate(s2, z2, z1))

    def _interpolate(self, s, z1, z2):
        row1, row2 = [], []
        for w1, w2 in zip(np.arange(0, 1, 0.1), np.arange(1, 0, -0.1)):
            sum = w1 * z1 + w2 * z2
            rec = self.reconstructor.predict([s, sum])[0, :, :, 0]
            if w1 < 0.5:
                row1.append(rec)
            else:
                row2.append(rec)
        return np.concatenate([np.concatenate(row1, axis=1), np.concatenate(row2, axis=1)], axis=0)
