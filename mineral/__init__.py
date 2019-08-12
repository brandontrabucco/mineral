"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf


def to_float(
    *args,
    **kwargs
):
    return (
        nested_apply(lambda x: tf.cast(x, tf.float32), args),
        nested_apply(lambda x: tf.cast(x, tf.float32), kwargs)
    )


def nested_apply(
    function,
    *structures
):
    if (isinstance(structures[0], np.ndarray) or 
            isinstance(structures[0], tf.Tensor) or not (
            isinstance(structures[0], list) or 
            isinstance(structures[0], tuple) or 
            isinstance(structures[0], set) or
            isinstance(structures[0], dict))):
        return function(*structures)
    elif isinstance(structures[0], list):
        return [
            nested_apply(
                function,
                *x,
            )
            for x in zip(*structures)
        ]
    elif isinstance(structures[0], tuple):
        return tuple(
            nested_apply(
                function,
                *x,
            )
            for x in zip(*structures)
        )
    elif isinstance(structures[0], set):
        return {
            nested_apply(
                function,
                *x,
            )
            for x in zip(*structures)
        }
    elif isinstance(structures[0], dict):
        keys_list = structures[0].keys()
        values_list = [[y[key] for key in keys_list] for y in structures]
        return {
            key: nested_apply(
                function,
                *values
            )
            for key, values in zip(keys_list, zip(*values_list))
        }


def discounted_sum(
    terms,
    discount_factor
):
    weights = tf.tile([[discount_factor]], [1, tf.shape(terms)[1]])
    weights = tf.math.cumprod(weights, axis=1, exclusive=True)
    return tf.math.cumsum(terms * weights, axis=1, reverse=True) / weights
