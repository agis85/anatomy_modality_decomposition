from skimage.measure import block_reduce
import numpy as np

import utils.image_utils, utils.data_utils
import logging
log = logging.getLogger('data')


class Data(object):

    def __init__(self, images, masks, index, scanner, downsample=1):
        """
        Data constructor.
        :param images:      a 4-D numpy array of images. Expected shape: (N, H, W, 1)
        :param masks:       a 4-D numpy array of myocardium segmentation masks. Expected shape: (N, H, W, 1)
        :param index:       a 1-D numpy array indicating the volume each image/mask belongs to. Used for data selection.
        """
        if images is None:
            raise ValueError('Images cannot be None.')
        if masks is None:
            raise ValueError('Masks cannot be None.')
        if index is None:
            raise ValueError('Index cannot be None.')
        if images.shape[:-1] != masks.shape[:-1]:
            raise ValueError('Image shape=%s different from Mask shape=%s' % (str(images.shape), str(masks.shape)))
        if images.shape[0] != index.shape[0]:
            raise ValueError('Different number of images and indices: %d vs %d' % (images.shape[0], index.shape[0]))

        self.images = images
        self.masks  = masks
        self.index  = index
        self.scanner = scanner
        self.num_volumes = len(self.volumes())

        self.downsample(downsample)

        log.info('Creating Data object with images of shape %s and %d volumes' % (str(images.shape), self.num_volumes))
        log.info('Images value range [%.1f, %.1f]' % (images.min(), images.max()))
        log.info('Masks value range [%.1f, %.1f]' % (masks.min(), masks.max()))

    def copy(self):
        return Data(np.copy(self.images), np.copy(self.masks), np.copy(self.index), np.copy(self.scanner))

    def merge(self, other):
        assert self.images.shape[1:] == other.images.shape[1:], str(self.images.shape) + ' vs ' + str(other.images.shape)
        assert self.masks.shape[1:] == other.masks.shape[1:], str(self.masks.shape) + ' vs ' + str(other.masks.shape)

        self.images = np.concatenate([self.images, other.images], axis=0)
        self.masks  = np.concatenate([self.masks, other.masks], axis=0)
        self.index  = np.concatenate([self.index, other.index], axis=0)
        self.scanner= np.concatenate([self.scanner, other.scanner], axis=0)
        self.num_volumes = len(self.volumes())
        log.info('Merged Data object of %d to this Data object of size %d' % (other.size(), self.size()))

    def select_masks(self, num_masks):
        log.info('Selecting the first %d masks out of %d.' % (num_masks, self.masks.shape[-1]))
        self.masks = self.masks[..., 0:num_masks]

    def crop(self, shape):
        log.debug('Cropping images and masks to shape ' + str(shape))
        [images], [masks] = utils.data_utils.crop_same([self.images], [self.masks], size=shape, pad_mode='constant')
        self.images = images
        self.masks  = masks
        assert self.images.shape[1:-1] == self.masks.shape[1:-1], \
            'Invalid shapes: ' + str(self.images.shape[1:-1]) + ' ' + str(self.masks.shape[1:-1])

    def volumes(self):
        return sorted(set(self.index))

    def get_images(self, vol):
        return self.images[self.index == vol]

    def get_masks(self, vol):
        return self.masks[self.index == vol]

    def get_scanner(self, vol):
        return self.scanner[self.index == vol]

    def filter_by_scanner(self, scanner):
        assert scanner in self.scanner, '%s is not a valid scanner type' % str(scanner)
        self.images  = self.images[self.scanner == scanner]
        self.masks   = self.masks[self.scanner == scanner]
        self.index   = self.index[self.scanner == scanner]
        self.scanner = self.scanner[self.scanner == scanner]
        self.num_volumes = len(self.volumes())
        log.debug('Selected %d volumes acquired with scanner %s' % (self.num_volumes, str(scanner)))

    def size(self):
        return len(self.images)

    def sample_per_volume(self, num, seed=-1):
        log.info('Sampling %d from each volume' % num)
        if seed > -1:
            np.random.seed(seed)

        new_images, new_masks, new_scanner, new_index = [], [], [], []
        for vol in self.volumes():
            images  = self.get_images(vol)
            masks   = self.get_masks(vol)
            scanner = self.get_scanner(vol)

            if images.shape[0] < num:
                log.debug('Volume %d contains less images: %d < %d. Sampling %d images.' %
                          (vol, images.shape[0], num, images.shape[0]))
                num = images.shape[0]

            idx = np.random.choice(images.shape[0], size=num, replace=False)
            images  = np.array([images[i] for i in idx])
            masks   = np.array([masks[i] for i in idx])
            scanner = np.array([scanner[i] for i in idx])
            index   = np.array([vol] * num)

            new_images.append(images)
            new_masks.append(masks)
            new_scanner.append(scanner)
            new_index.append(index)

        self.images  = np.concatenate(new_images, axis=0)
        self.masks   = np.concatenate(new_masks, axis=0)
        self.scanner = np.concatenate(new_scanner, axis=0)
        self.index   = np.concatenate(new_index, axis=0)

        log.info('Sampled %d images.' % len(self.images))

    def sample_images(self, num, seed=-1):
        log.info('Sampling %d images out of total %d' % (num, self.size()))
        if seed > -1:
            np.random.seed(seed)

        idx = np.random.choice(self.size(), size=num, replace=False)
        self.images  = np.array([self.images[i] for i in idx])
        self.masks   = np.array([self.masks[i] for i in idx])  # self.masks[:num]
        self.scanner = np.array([self.scanner[i] for i in idx])
        self.index   = np.array([self.index[i] for i in idx])

    def sample(self, num, seed=-1):
        log.info('Sampling %d volumes out of total %d' % (num, self.num_volumes))
        if seed > -1:
            np.random.seed(seed)

        if num == self.num_volumes:
            return

        volumes = np.random.choice(self.volumes(), size=num, replace=False)
        if num == 0 or len(volumes) == 0:
            self.images  = np.zeros(shape=(0,) + self.images.shape[1:])
            self.masks   = np.zeros(shape=(0,) + self.masks.shape[1:])
            self.scanner = np.zeros(shape=(0,) + self.scanner.shape[1:])
            self.index   = np.zeros(shape=(0,) + self.index.shape[1:])
            self.num_volumes = 0
            return

        self.images  = np.concatenate([self.get_images(v) for v in volumes], axis=0)
        self.masks   = np.concatenate([self.get_masks(v) for v in volumes], axis=0)
        self.scanner = np.concatenate([self.get_scanner(v) for v in volumes], axis=0)
        self.index   = np.concatenate([self.index.copy()[self.index == v] for v in volumes], axis=0)
        self.num_volumes = len(volumes)

        log.info('Sampled volumes: %s of total %d images' % (str(volumes), self.size()))

    def shape(self):
        return self.images.shape

    def downsample(self, ratio=2):
        if ratio == 1: return

        self.images = block_reduce(self.images, block_size=(1, ratio, ratio, 1), func=np.mean)
        if self.masks is not None:
            self.masks  = block_reduce(self.masks, block_size=(1, ratio, ratio, 1), func=np.mean)

        log.info('Downsampled data by %d to shape %s' % (ratio, str(self.images.shape)))

    def get_lvv(self, slice_thickness, pixel_resolution):
        lv = self.masks[..., 1:2]
        return lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution

    def get_lvv_per_slice(self, vol_i, slice_thickness, pixel_resolution):
        masks = self.get_masks(vol_i)
        lv = masks[..., 1:2]
        return lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution

    def get_lvv_per_volume(self, vol_i, slice_thickness, pixel_resolution):
        masks = self.get_masks(vol_i)
        lv = masks[..., 1:2]
        return np.sum(lv.sum(axis=(1, 2, 3)) * slice_thickness * pixel_resolution)
