"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf


def conjugate_gradient(
        matrix_vector_product_function,
        initial_guess,
        target,
        tolerance=1e-3,
        maximum_iterations=100
):
    x = initial_guess
    Ax = matrix_vector_product_function(x)
    xAx = tf.reduce_sum([
        tf.reduce_sum(x_i * Ax_i)
        for x_i, Ax_i in zip(x, Ax)
    ])
    r = [
        target_i - Ax_i
        for target_i, Ax_i in zip(target, Ax)
    ]
    rTr = tf.reduce_sum([
        tf.reduce_sum(r1_i * r2_i)
        for r1_i, r2_i in zip(r, r)
    ])
    p = r
    for i in range(maximum_iterations):
        if rTr < tolerance:
            break
        Ap = matrix_vector_product_function(p)
        pAp = tf.reduce_sum([
            tf.reduce_sum(p_i * Ap_i)
            for p_i, Ap_i in zip(p, Ap)
        ])
        alpha = rTr / pAp
        x = [
            x_i + alpha * p_i
            for x_i, p_i in zip(x, p)
        ]
        xAx += rTr
        r = [
            r_i - alpha * Ap_i
            for r_i, Ap_i in zip(r, Ap)
        ]
        rTr_next = tf.reduce_sum([
            tf.reduce_sum(r1_i * r2_i)
            for r1_i, r2_i in zip(r, r)
        ])
        beta = rTr_next / rTr
        rTr = rTr_next
        p = [
            r_i + beta * p_i
            for r_i, p_i in zip(r, p)
        ]
    return x, xAx
