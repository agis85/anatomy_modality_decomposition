import logging

import numpy as np
from keras import backend as K

log = logging.getLogger()


def dice(y_true, y_pred, binarise=False, smooth=0.1):
    y_pred = y_pred[..., 0:y_true.shape[-1]]

    # Cast the prediction to binary 0 or 1
    if binarise:
        y_pred = np.round(y_pred)

    # Symbolically compute the intersection
    y_int = y_true * y_pred
    return np.mean((2 * np.sum(y_int, axis=(1, 2, 3)) + smooth)
                   / (np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3)) + smooth))


def dice_coef(y_true, y_pred):
    '''
    DICE Loss.
    :param y_true: a tensor of ground truth data
    :param y_pred: a tensor of predicted data
    '''
    # Symbolically compute the intersection
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3)) + 0.1
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) + 0.1
    return K.mean(2 * intersection / union, axis=0)


# Technically this is the negative of the Sorensen-Dice index. This is done for minimization purposes
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def make_dice_loss_fnc(restrict_chn=1):
    log.debug('Making DICE loss function for the first %d channels' % restrict_chn)

    def dice_fnc(y_true, y_pred):
        y_pred_new = y_pred[..., 0:restrict_chn] + 0.
        intersection = K.sum(y_true * y_pred_new, axis=(1, 2, 3))
        union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred_new, axis=(1, 2, 3)) + 0.1
        return 1 - K.mean(2 * (intersection + 0.1) / union, axis=0)

    return dice_fnc


def kl(args):
    mean, log_var = args
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return K.reshape(kl_loss, (-1, 1))


def ypred(y_true, y_pred):
    return y_pred
