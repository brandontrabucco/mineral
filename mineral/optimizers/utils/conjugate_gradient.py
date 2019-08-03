"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def dot(
    x,
    y
):
    return tf.reduce_sum([
        tf.reduce_sum(x_i * y_i)
        for x_i, y_i in zip(x, y)
    ])


def conjugate_gradient(
    matrix_vector_product_function,
    initial_guess,
    target,
    tolerance=1e-3,
    maximum_iterations=100
):
    x = initial_guess
    Ax = matrix_vector_product_function(x)
    r = [
        target_i - Ax_i
        for target_i, Ax_i in zip(target, Ax)
    ]
    rTr = dot(r, r)
    p = r
    for i in range(maximum_iterations):
        if rTr < tolerance:
            break
        Ap = matrix_vector_product_function(p)
        pAp = dot(p, Ap)
        alpha = rTr / pAp
        x = [
            x_i + alpha * p_i
            for x_i, p_i in zip(x, p)
        ]
        r = [
            r_i - alpha * Ap_i
            for r_i, Ap_i in zip(r, Ap)
        ]
        rTr_next = dot(r, r)
        beta = rTr_next / rTr
        rTr = rTr_next
        p = [
            r_i + beta * p_i
            for r_i, p_i in zip(r, p)
        ]
    Ax = matrix_vector_product_function(x)
    xAx = dot(x, Ax)
    return x, xAx
