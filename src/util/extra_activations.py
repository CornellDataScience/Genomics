from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def binary_step(x):
    return 1 if x else 0


def block(x):
    return K.ones(K.backend.shape(x))


get_custom_objects().update({
    'binary_step': Activation(binary_step),
    'block': Activation(block)
})
