"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.conjugate_gradient import conjugate_gradient


def fisher_vector_product(
    forward_function,
    hessian_function,
    trainable_variables,
    y
):
    with tf.GradientTape(persistent=True) as tape:
        x = forward_function()
        v = [
            tf.ones(tf.shape(x_i))
            for x_i in x
        ]
        for v_i in v:
            tape.watch(v_i)
        g = [
            tape.gradient(
                x_i,
                trainable_variables,
                output_gradients=v_i
            )
            for x_i, v_i in zip(x, v)
        ]
    h = hessian_function(*x)
    jvp = [
        h_i * tape.gradient(
            g_i,
            v_i,
            output_gradients=y
        )
        for h_i, g_i, v_i in zip(h, g, v)
    ]
    fvp = [
        tape.gradient(
            x_i,
            trainable_variables,
            output_gradients=jvp_i
        )
        for x_i, jvp_i in zip(x, jvp)
    ]
    for fvp_i in fvp[1:]:
        for j in range(len(fvp_i)):
            fvp[0][j] += fvp_i[j]
    return fvp[0]


def inverse_fisher_vector_product(
    forward_function,
    hessian_function,
    trainable_variables,
    y,
    tolerance=1e-3,
    maximum_iterations=100
):
    return conjugate_gradient(
        lambda x: fisher_vector_product(
            forward_function,
            hessian_function,
            trainable_variables,
            x
        ),
        y,
        y,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations
    )
