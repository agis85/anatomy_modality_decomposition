'''
This file contains a keras Regularizer that can be used to encourage the largest sigular
value (lsv) of either a Dense or a Conv2D layer to be <= 1.

For Dense layers the linear transformation they perform is a multiplication by the
weight matrix W (and then the addition of a bias term, but this does not affect the gradient
so we can ignore it). So, for a Dense layer the lsv is just: lsv(W)

For Conv2D layers things are more complicated. The weight matrix W does define the
transformation, but it isn't defined simply as multiplication of the input by W, rather,
the input is multiplied by a matrix that is some function of W (and the size of the
input image, which we will call im_sz). Lets call the suitable function frmr, then for
a Conv2D layer the lsv is: lsv(frmr(W,im_sz)).

(As an asside, in the spectral normalization paper they seem to directly use lsv(W) even
in the Conv2D case. I think this is technically wrong, but has a sort of similar result,
in that it does provide some pressure for lsv(frmr(W,im_sz)) to be low-ish, but making
lsv(W) <= 1 doesn't neccessarily imply lsv(frmr(W,im_sz)) <= 1)

We need to define both lsv and frmr in a differentiable way so they can be used to
train a network with backprop. The spectral normalization paper proposes a method to
approximate lsv in a differentiable way (which we will come back to later), so we just
need to work out a method for frmr.

Lets assume that the Conv2D layer has padding and a stride of 1, and also assume the
input is a single channel image, and we only have 1 filter. This means that the layer
maps (im_sz, im_sz, 1) --> (im_sz, im_sz, 1). Lets define n = im_sz * imsz, then
the layer defines a linear map from n-dimensional space to n-dimensional space. So, we
can flatten the image to an n-dimensional vector, then multiply it by some n by n
matrix, M, to get the n-dimensional vector output, which can be then reshaped into the
output image. Thus M = frmr(W,im_sz) (and so lsv(M) = lsv(frmr(W,im_sz)) ).

Specifically, M is made by arranging (and duplicating) the values of W into an n by n
matrix. we do this in the function make_M().

frmr(W,im_sz) = f(W)*g(im_sz)

a,b,c,d
e,f,g,h
i,j,k,l
m,n,o,p

'''
from keras import backend as K
from keras.regularizers import Regularizer
import numpy as np


class Spectral(Regularizer):
    ''' Spectral normalization regularizer
        # Arguments
            alpha = weight for regularization penalty
    '''

    def __init__(self, dim, alpha=K.variable(10.)):
        '''
        in a Conv2D layer dim needs to be num_channels in the previous layer times the filter_size^2
        in a Dense layer dim needs to be num_channels in the previous layer
        '''

        self.dim = dim
        self.alpha = alpha  # K.cast_to_floatx(alpha)
        self.u = K.variable(np.random.random((dim, 1)) * 2 - 1.)

    def __call__(self, x):
        x_shape = K.shape(x)
        x = K.reshape(x, (-1, x_shape[-1]))  # this deals with convolutions, fingers crossed!

        for itters in range(3):
            WTu = K.dot(K.transpose(x), self.u)
            v = WTu / K.sqrt(K.sum(K.square(WTu)))

            Wv = K.dot(x, v)
            self.u = Wv / K.sqrt(K.sum(K.square(Wv)))

        spectral_norm = K.dot(K.dot(K.transpose(self.u), x), v)

        target_x = K.stop_gradient(x / spectral_norm)
        return self.alpha * K.mean(K.abs(target_x - x))

    def get_config(self):
        return {'alpha': float(self.alpha)}
