"""Where alchemy, heresical practices, black magic, and witch spells happen.

TODO: Border values are often classified incorrectly.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


def to_binary(x):
    # Don't mess with inputs.
    x = x.copy().astype('float32')
    if 0. in x:
        return (x != 0).astype('float32')
    return (x == x[0]).astype('float32')


def to_categorical(x, return_index=False):
    """Make a vector categorical.

    tf.keras.utils.to_categorical is stupid. It treats 2 and 2.1 as
    different classes, yet they're encoded as the same category. And it
    doesn't even support strings.
    """
    x = x.copy()
    index = []
    for i, item in enumerate(x):
        if item not in index:
            index.append(item)
        x[i] = index.index(item)
    x = tf.keras.utils.to_categorical(x, num_classes=len(index))

    if return_index:
        return x, index
    else:
        return x


def normalize(x, return_mean=False, return_std=False):
    x = x.copy().astype('float32')
    mean = x.mean(axis=0)
    std = x.std(axis=0)

    returns = [(x - mean) / std]
    if return_mean:
        returns.append(mean)
    if return_std:
        returns.append(std)
    return returns[0] if len(returns) == 1 else tuple(returns)


def preprocess(x, return_index=False):
    """Make vector ``x`` good.

    If appropriate, make ``x`` binary or one-hot etc. Make ``x`` the
    right dtype.

    It doesn't normalize ``x``. Use ``normalize``.
    """
    # TODO, support matrix preprocessing.
    x = x.copy()
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.ravel()
    if x.ndim != 1:
        raise ValueError(f'Unsupported of dimensions of {x}')
    classes = []

    x_type = vector_type(x)
    if x_type == 'binary':
        x = to_binary(x)
    elif x_type == 'multiclass':
        if return_index:
            x, classes = to_categorical(x, return_index=True)
        else:
            x = to_categorical(x, return_index=False)
    elif x_type == 'regression':
        x = x.astype('float32')
    else:
        raise NotImplementedError(f'{x_type} is not supported.')

    if return_index:
        return x, classes
    else:
        return x


def vector_type(x):
    classes = np.unique(x)
    if len(classes) == 2:
        return 'binary'
    elif 2 < len(classes) <= len(x) / 3:
        return 'multiclass'
    elif type(x[0]) in {bool, float, int, complex}:
        return 'regression'
    elif type(x[0]) in {tuple, set, dict, list}:
        return 'list'
    elif type(x[0]) in {str, bytes, bytearray}:
        return 'text'


def load_model(input_shape, model_type, num_classes=None):
    if model_type == 'binary':
        return binary_model(input_shape=input_shape)
    if model_type == 'multiclass':
        assert num_classes, 'Specify argument ``num_classes`` for multiclass.'
        return multiclass_model(input_shape=input_shape,
                                num_classes=num_classes)
    elif model_type == 'regression':
        return regression_model(input_shape=input_shape)
    elif model_type == 'list':
        raise NotImplementedError
        return None
    elif model_type == 'text':
        raise NotImplementedError
        return None


def binary_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(Dense(6, input_shape=input_shape))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(lr=1e-3))

    return model


def multiclass_model(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(Dense(6, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(6))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(lr=1e-3))

    return model


def regression_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(Dense(6, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(Dropout(0.3))
    model.add(Dense(6))
    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(loss='mse',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam(lr=1e-3))

    return model
