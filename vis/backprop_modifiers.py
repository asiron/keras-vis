from __future__ import absolute_import

from . import backend
from .utils import utils


def guided(model, custom_objects=None):
    """Modifies backprop to only propagate positive gradients for positive activations.

    Args:
        model: The `keras.models.Model` instance whose gradient computation needs to be overridden.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    References:
        Details on guided back propagation can be found in paper: [String For Simplicity: The All Convolutional Net]
        (https://arxiv.org/pdf/1412.6806.pdf)
    """
    return backend.modify_model_backprop(model, 'guided', custom_objects=custom_objects)


def rectified(model, custom_objects=None):
    """Modifies backprop to only propagate positive gradients.

    Args:
        model: The `keras.models.Model` instance whose gradient computation needs to be overridden.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    References:
        Details can be found in the paper: [Visualizing and Understanding Convolutional Networks]
        (https://arxiv.org/pdf/1311.2901.pdf)
    """
    return backend.modify_model_backprop(model, 'rectified', custom_objects=custom_objects)


# Create aliases
relu = deconv = rectified


def get(identifier):
    return utils.get_identifier(identifier, globals(), __name__)
