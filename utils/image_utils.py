
import numpy as np
import os
import scipy
from PIL import Image, ImageDraw
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_fill_holes
import utils.data_utils


def save_multiimage_segmentation(x, m, y, folder, epoch):
        rows = []
        for i in range(x.shape[0]):
            y_list = [y[i, :, :, chn] for chn in range(y.shape[-1])]
            m_list = [m[i, :, :, chn] for chn in range(m.shape[-1])]
            if m.shape[-1] < y.shape[-1]:
                m_list += [np.zeros(shape=(m.shape[1], m.shape[2]))] * (y.shape[-1] - m.shape[-1])
            assert len(y_list) == len(m_list), 'Incompatible sizes: %d vs %d' % (len(y_list), len(m_list))
            rows += [np.concatenate([x[i, :, :, 0]] + y_list + m_list, axis=1)]

        im_plot = np.concatenate(rows, axis=0)
        scipy.misc.imsave(folder + '/segmentations_epoch_%d.png' % (epoch), im_plot)


def save_segmentation(folder, model, images, masks, name_prefix):
    '''
    :param folder: folder to save the image
    :param model : segmentation model
    :param images: an image of shape [H,W,chn]
    :param masks : a mask of shape [H,W,chn]
    :return      : the predicted segmentation mask
    '''
    images = np.expand_dims(images, axis=0)
    masks  = np.expand_dims(masks, axis=0)
    s = model.predict(images)

    # In this case the segmentor is multi-output, with each output corresponding to a mask.
    if len(s[0].shape) == 4:
        s = np.concatenate(s, axis=-1)

    mask_list_pred = [s[:, :, :, j:j + 1] for j in range(s.shape[-1])]
    mask_list_real = [masks[:, :, :, j:j + 1] for j in range(masks.shape[-1])]
    if masks.shape[-1] < s.shape[-1]:
        mask_list_real += [np.zeros(shape=masks.shape[0:3] + (1,))] * (s.shape[-1] - masks.shape[-1])

    # if we use rotations, the sizes might differ
    m1, m2 = utils.data_utils.crop_same(mask_list_real, mask_list_pred)
    images_cropped, _ = utils.data_utils.crop_same([images], [images.copy()], size=(m1[0].shape[1], m1[0].shape[2]))
    mask_list_real = [s[0, :, :, 0] for s in m1]
    mask_list_pred = [s[0, :, :, 0] for s in m2]
    images_cropped = [s[0, :, :, 0] for s in images_cropped]

    row1 = np.concatenate(images_cropped + mask_list_pred, axis=1)
    row2 = np.concatenate(images_cropped + mask_list_real, axis=1)
    im = np.concatenate([row1, row2], axis=0)
    imsave(os.path.join(folder, name_prefix + '.png'), im)
    return s


def convert_myo_to_lv(mask):
    '''
    Create a LV mask from a MYO mask. This assumes that the MYO is connected.
    :param mask: a 4-dim myo mask
    :return:     a 4-dim array with the lv mask.
    '''
    assert len(mask.shape) == 4, mask.shape

    # If there is no myocardium, then there's also no LV.
    if mask.sum() == 0:
        return np.zeros(mask)

    assert mask.max() == 1 and mask.min() == 0

    mask_lv = []
    for slc in range(mask.shape[0]):
        myo = mask[slc, :, :, 0]
        myo_lv = binary_fill_holes(myo).astype(int)
        lv = myo_lv - myo
        mask_lv.append(np.expand_dims(np.expand_dims(lv, axis=0), axis=-1))
    return np.concatenate(mask_lv, axis=0)


def makeTextHeaderImage(col_widths, headings, padding=(5, 5)):
    im_width = len(headings) * col_widths
    im_height = padding[1] * 2 + 11

    img = Image.new('RGB', (im_width, im_height), (0, 0, 0))
    d = ImageDraw.Draw(img)

    for i, txt in enumerate(headings):

        while d.textsize(txt)[0] > col_widths - padding[0]:
            txt = txt[:-1]
        d.text((col_widths * i + padding[0], + padding[1]), txt, fill=(1, 0, 0))

    raw_img_data = np.asarray(img, dtype="int32")

    return raw_img_data[:, :, 0]


def get_roi_dims(mask_list, size_mult=16):
    # This assumes each element in the mask list has the same dimensions
    masks = np.concatenate(mask_list, axis=0)
    masks = np.squeeze(masks)
    assert len(masks.shape) == 3

    lx, hx, ly, hy = 0, 0, 0, 0
    for y in range(masks.shape[2] - 1, 0, -1):
        if masks[:, :, y].max() == 1:
            hy = y
            break
    for y in range(masks.shape[2]):
        if masks[:, :, y].max() == 1:
            ly = y
            break
    for x in range(masks.shape[1] - 1, 0, -1):
        if masks[:, x, :].max() == 1:
            hx = x
            break
    for x in range(masks.shape[1]):
        if masks[:, x, :].max() == 1:
            lx = x
            break

    l = np.max([np.min([lx, ly]) - 10, 0])
    r = np.min([np.max([hx, hy]) + 10, masks.shape[2]])

    l, r = greatest_common_divisor(l, r, size_mult)

    return l, r


def greatest_common_divisor(l, r, size_mult):
    if (r - l) % size_mult != 0:
        div = (r - l) / size_mult
        if div * size_mult < (div + 1) * size_mult:
            diff = (r - l) - div * size_mult
            l += diff / 2
            r -= diff - (diff / 2)
        else:
            diff = (div + 1) * size_mult - (r - l)
            l -= diff / 2
            r += diff - (diff / 2)
    return int(l), int(r)