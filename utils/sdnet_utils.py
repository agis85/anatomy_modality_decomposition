
import keras.backend as K
import numpy as np

from utils.distributions import NormalDistribution


def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Instead of sampling from Q(z|X), sample eps = N(0,I): z = z_mean + sqrt(var)*eps
    :param args: args (tensor): mean and log of variance of Q(z|X)
    :return:     z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_sample(args):
    z_mean, z_log_var = args
    batch = z_mean.shape[0]
    dim = z_mean.shape[1]
    # by default, random_normal has mean=0 and std=1.0
    gaussian = NormalDistribution()
    epsilon = gaussian.sample((batch, dim))
    return z_mean + np.exp(0.5 * z_log_var) * epsilon


def get_net(trainer_model, name):
    layers = [l for l in trainer_model.layers if l.name == name]
    assert len(layers) == 1
    return layers[0]


def make_trainable(model, val):
    """
    Helper method to enable/disable training of a model
    :param model: a Keras model
    :param val:   True/False
    """
    model.trainable = val
    try:
        for l in model.layers:
            try:
                for k in l.layers:
                    make_trainable(k, val)
            except:
                # Layer is not a model, so continue
                pass
            l.trainable = val
    except:
        # Layer is not a model, so continue
        pass