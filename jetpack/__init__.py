"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf


def flat_multiply(
    a,
    b
):
    a = tf.concat([tf.reshape(x, [-1]) for x in a], 0)
    b = tf.concat([tf.reshape(x, [-1]) for x in b], 0)
    return tf.reduce_sum(a * b)


def conjugate_gradient(
        matrix_vector_product_function,
        initial_guess,
        target,
        tolerance=1e-3,
        maximum_iterations=100
):
    x = initial_guess
    Ax = matrix_vector_product_function(x)
    r = [target_i - Ax_i for target_i, Ax_i in zip(target, Ax)]
    p = r
    for i in range(maximum_iterations):
        rTr = flat_multiply(r, r)
        if rTr < tolerance:
            break
        Ap = matrix_vector_product_function(p)
        pAp = flat_multiply(p, Ap)
        alpha = rTr / pAp
        x = [x_i + alpha * p_i for x_i, p_i in zip(x, p)]
        r = [r_i - alpha * Ap_i for r_i, Ap_i in zip(r, Ap)]
        beta = flat_multiply(r, r) / rTr
        p = [r_i + beta * p_i for r_i, p_i in zip(r, p)]
    return x


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
