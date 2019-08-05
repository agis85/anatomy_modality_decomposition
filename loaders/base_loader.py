
import os
import numpy as np
from abc import abstractmethod


class Loader(object):
    """
    Abstract class defining the behaviour of loaders for different datasets.
    """
    def __init__(self):
        self.num_masks   = 0
        self.num_volumes = 0
        self.input_shape = (None, None, 1)
        self.data_folder = None
        self.volumes = sorted(self.splits()[0]['training'] +
                              self.splits()[0]['validation'] +
                              self.splits()[0]['test'])
        self.log = None

    @abstractmethod
    def splits(self):
        """
        :return: an array of splits into validation, test and train indices
        """
        pass

    @abstractmethod
    def load_labelled_data(self, split, split_type, modality, normalise=True, value_crop=True, downsample=1):
        """
        Load labelled data from saved numpy arrays.
        Assumes a naming convention of numpy arrays as:
        <dataset_name>_images.npz, <dataset_name>_masks_lv.npz, <dataset_name>_masks_myo.npz etc.
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.

        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :param downsample:  downsample image ratio - used for for testing
        :return:            a Data object containing the loaded data
        """
        pass

    @abstractmethod
    def load_unlabelled_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        """
        Load unlabelled data from saved numpy arrays.
        Assumes a naming convention of numpy arrays as ul_<dataset_name>_images.npz
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :return:            a Data object containing the loaded data
        """
        pass

    @abstractmethod
    def load_all_data(self, split, split_type, modality='MR', normalise=True, value_crop=True):
        """
        Load all images (labelled and unlabelled) from saved numpy arrays.
        Assumes a naming convention of numpy arrays as all_<dataset_name>_images.npz
        If numpy arrays are not found, then data is loaded from sources and saved in numpy arrays.
        :param split:       the split number, e.g. 0, 1
        :param split_type:  the split type, e.g. training, validation, test, all (for all data)
        :param modality:    modality to load if the dataset has multimodal data
        :param normalise:   True/False: normalise images to [-1, 1]
        :param value_crop:  True/False: crop values between 5-95 percentiles
        :return:            a Data object containing the loaded data
        """
        pass

    @abstractmethod
    def load_raw_labelled_data(self, normalise=True, value_crop=True):
        """
        Load raw data, do preprocessing e.g. normalisation, resampling, value cropping etc
        :param normalise:  True or False to normalise data
        :param value_crop: True or False to crop in the 5-95 percentiles or not.
        :return:           a pair of arrays (images, index)
        """
        pass

    @abstractmethod
    def load_raw_unlabelled_data(self, include_labelled, normalise=True, value_crop=True):
        """
        Load raw data, do preprocessing e.g. normalisation, resampling, value cropping etc
        :param normalise:  True or False to normalise data
        :param value_crop: True or False to crop in the 5-95 percentiles or not.
        :return:           a pair of arrays (images, index)
        """

    def base_load_unlabelled_images(self, dataset, split, split_type, include_labelled, normalise, value_crop):
        """
        Load only images.
        :param dataset:             dataset name
        :param split:               the split number, e.g. 0, 1
        :param split_type:          the split type, e.g. training, validation, test, all (for all data)
        :param include_labelled:    include images from ES, ED phases that are labelled. Can be True/False
        :param normalise:           True or False to normalise data
        :param value_crop:          True or False to crop in the 5-95 percentiles or not.
        :return:                    a tuple of images and index arrays.
        """
        npz_prefix_type = 'ul_' if not include_labelled else 'all_'
        npz_prefix = npz_prefix_type + 'norm_' if normalise else npz_prefix_type + 'unnorm_'

        # Load saved numpy array
        if os.path.exists(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz')):
            images = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_images.npz'))['arr_0']
            index  = np.load(os.path.join(self.data_folder, npz_prefix + dataset + '_index.npz'))['arr_0']
            self.log.debug('Loaded compressed ' + dataset + ' unlabelled data of shape ' + str(images.shape))
        # Load from source
        else:
            images, index = self.load_raw_unlabelled_data(include_labelled, normalise, value_crop)
            images = np.expand_dims(images, axis=3)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_images'), images)
            np.savez_compressed(os.path.join(self.data_folder, npz_prefix + dataset + '_index'), index)

        assert split_type in ['training', 'validation', 'test', 'all'], 'Unknown split_type: ' + split_type

        if split_type == 'all':
            return images, index

        volumes = self.splits()[split][split_type]
        images = np.concatenate([images[index == v] for v in volumes])
        index  = np.concatenate([index[index==v] for v in volumes])
        return images, index