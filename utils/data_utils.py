import logging
import numpy as np

log = logging.getLogger('data_utils')


def normalise(array, min_value, max_value):
    """
    Rescale an array in the min and max value defined
    :param array:       array to process
    :param min_value:   new minimum
    :param max_value:   new maximum
    :return:            rescaled array
    """
    array = (max_value - min_value) * (array - float(array.min())) / (array.max() - array.min()) + min_value
    assert array.max() == max_value and array.min() == min_value
    return array


def generator(batch, seed, mode, *x):
    """
    Create a data iterator
    :param batch: batch size
    :param seed:  numpy seed
    :param mode:  can be one of ['overflow', 'no_overflow']. If the last batch is smaller than the batch size, the mode
                  defines if the iterator will overflow to fill the last batch.
    :param x:     the input arrays
    :return:      the iterator
    """
    assert mode in ['overflow', 'no_overflow']
    imshape = x[0].shape
    for ar in x:
        # case where all inputs are images
        if len(ar.shape) == len(imshape):
            assert ar.shape[:-1] == imshape[:-1], str(ar.shape) + ' vs ' + str(imshape)
        # case where inputs might be arrays of different dimensions
        else:
            assert ar.shape[0] == imshape[0], str(ar.shape) + ' vs ' + str(imshape)

    start = 0
    while 1:
        if isempty(*x):  # if the arrays are empty do not process and yield empty arrays
            log.info('Empty inputs. Return empty arrays')
            res = []
            for ar in x:
                res.append(np.empty(shape=ar.shape))
            if len(res) > 1:
                yield res
            else:
                yield res[0]
        else:
            start, ims = generate(start, batch, seed, mode, *x)
            if len(ims) == 1:
                yield ims[0]
            else:
                yield ims


def isempty(*x):
    for ar in x:
        if ar.shape[0] > 0:
            return False
    return True


def generate(start, batch, seed, mode, *images):
    np.random.seed(seed)

    result = []

    if mode == 'no_overflow':
        for ar in images:
            result.append(ar[start:start + batch])
        start += batch

        if start >= len(images[0]):
            index = np.array(range(len(images[0])))
            np.random.shuffle(index)
            for ar in images:
                ar[:] = ar[index]  # shuffle array
            start = 0

        return start, result

    if start + batch <= len(images[0]):
        for ar in images:
            result.append(ar[start:start + batch])
        start += batch
        return start, result
    else:
        # shuffle images
        index = np.array(range(len(images[0])))
        np.random.shuffle(index)

        extra = batch + start - len(images[0])  # extra images to use from the beginning
        for ar in images:
            ims = ar[start:]  # last images of array
            ar[:] = ar[index]  # shuffle array
            if extra > 0:
                result.append(np.concatenate([ims, ar[0:extra]], axis=0))

        return extra, result


def crop_same(image_list, mask_list, size=(None, None), mode='equal', pad_mode='edge'):
    '''
    Crop the data in the image and mask lists, so that they have the same size.
    :param image_list: a list of images. Each element should be 4-dimensional, (sl,h,w,chn)
    :param mask_list:  a list of masks. Each element should be 4-dimensional, (sl,h,w,chn)
    :param size:       dimensions to crop the images to.
    :param mode:       can be one of [equal, left, right]. Denotes where to crop pixels from. Defaults to middle.
    :param pad_mode:   can be one of ['edge', 'constant']. 'edge' pads using the values of the edge pixels,
                       'constant' pads with a constant value
    :return:           the modified arrays
    '''
    min_w = np.min([m.shape[1] for m in mask_list]) if size[0] is None else size[0]
    min_h = np.min([m.shape[2] for m in mask_list]) if size[1] is None else size[1]

    # log.debug('Resizing list1 of size %s to size %s' % (str(image_list[0].shape), str((min_w, min_h))))
    # log.debug('Resizing list2 of size %s to size %s' % (str(mask_list[0].shape), str((min_w, min_h))))

    img_result, msk_result = [], []
    for i in range(len(mask_list)):
        im = image_list[i]
        m = mask_list[i]

        if m.shape[1] > min_w:
            m = _crop(m, 1, min_w, mode)
        if im.shape[1] > min_w:
            im = _crop(im, 1, min_w, mode)
        if m.shape[1] < min_w:
            m = _pad(m, 1, min_w, pad_mode)
        if im.shape[1] < min_w:
            im = _pad(im, 1, min_w, pad_mode)

        if m.shape[2] > min_h:
            m = _crop(m, 2, min_h, mode)
        if im.shape[2] > min_h:
            im = _crop(im, 2, min_h, mode)
        if m.shape[2] < min_h:
            m = _pad(m, 2, min_h, pad_mode)
        if im.shape[2] < min_h:
            im = _pad(im, 2, min_h, pad_mode)

        img_result.append(im)
        msk_result.append(m)
    return img_result, msk_result


def _crop(image, dim, nb_pixels, mode):
    diff = image.shape[dim] - nb_pixels
    if mode == 'equal':
        l = int(np.ceil(diff / 2))
        r = image.shape[dim] - l
    elif mode == 'right':
        l = 0
        r = nb_pixels
    elif mode == 'left':
        l = diff
        r = image.shape[dim]
    else:
        raise 'Unexpected mode: %s. Expected to be one of [equal, left, right].' % mode

    if dim == 1:
        return image[:, l:r, :, :]
    elif dim == 2:
        return image[:, :, l:r, :]
    else:
        return None


def _pad(image, dim, nb_pixels, mode='edge'):
    diff = nb_pixels - image.shape[dim]
    l = int(diff / 2)
    r = int(diff - l)
    if dim == 1:
        pad_width = ((0, 0), (l, r), (0, 0), (0, 0))
    elif dim == 2:
        pad_width = ((0, 0), (0, 0), (l, r), (0, 0))
    else:
        return None

    if mode == 'edge':
        new_image = np.pad(image, pad_width, 'edge')
    elif mode == 'constant':
        new_image = np.pad(image, pad_width, 'constant', constant_values=np.min(image))
    else:
        raise Exception('Invalid pad mode: ' + mode)

    return new_image


def sample(data, nb_samples, seed=-1):
    if seed > -1:
        np.random.seed(seed)
    idx = np.random.choice(len(data), size=nb_samples, replace=False)
    return np.array([data[i] for i in idx])


def swap_last_columns(array, col1, col2):
    '''
    Swap the values between two columns in the last dimension of the array.
    :param array:   input array.
    :param col1:    index of column1
    :param col2:    index of column2
    :return:        the processed array
    '''

    temp = array[..., col1].copy()
    array[..., col1] = array[..., col2].copy()
    array[..., col2] = temp.copy()
    return array