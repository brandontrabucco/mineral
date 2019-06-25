"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf


def jacobian_vector_product(
    ys,
    xs,
    d_xs
):
    v = tf.placeholder(ys.dtype, shape=tf.shape(ys.get_shape()))
    g = tf.gradients(ys, xs, grad_ys=v)
    return tf.gradients(g, v, grad_ys=d_xs)


def to_float(
    *args,
    **kwargs
):
    cast_function = lambda x: tf.cast(x, tf.float32)
    return (
        nested_apply(cast_function, args),
        nested_apply(cast_function, kwargs)
    )


def flatten(
    x
):
    return tf.reshape(
        x, 
        (x.shape[0], -1),
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
        keys_list = [structures[0].keys()]
        values_list = [y.values() for y in structures]
        merged_list = keys_list + values_list
        return {
            key: nested_apply(
                function,
                *values,
            )
            for key, *values in zip(*merged_list)
        }
