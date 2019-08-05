
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
from tensorflow.python.framework import ops


class Rounding(Layer):
    """
    Custom layer that rounds a tensor.
    """
    def __init__(self, **kwargs):
        super(Rounding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rounding, self).build(input_shape)

    def call(self, x, **kwargs):
        return roundWithGrad(x)

    def compute_output_shape(self, input_shape):
        return input_shape


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))  # generate a unique name to avoid duplicates
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        res = tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        res[0].set_shape(inp[0].get_shape())
        return res


def roundWithGrad(x, name=None):
    with ops.name_scope(name, "roundWithGrad", [x]) as name:
        round_x = py_func(lambda x: np.round(x).astype('float32'), [x], [tf.float32], name=name,
                          grad=_roundWithGrad_grad)  # <-- here's the call to the gradient
        return round_x[0]


def _roundWithGrad_grad(op, grad):
    x = op.inputs[0]
    return grad * 1  # do whatever with gradient here (e.g. could return grad * 2 * x  if op was f(x)=x**2)
