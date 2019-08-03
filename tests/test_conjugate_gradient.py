"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.optimizers.utils.conjugate_gradient import conjugate_gradient


if __name__ == "__main__":

    matrix = tf.constant([
        [4.0, 1.0],
        [1.0, 3.0]
    ])

    def matrix_vector_product_function(vector):
        return [tf.linalg.matmul(vi, matrix) for vi in vector]

    initial_guess = [tf.constant([[2.0, 1.0]])]
    target = [tf.constant([[1.0, 2.0]])]

    result, xAx = conjugate_gradient(
        matrix_vector_product_function,
        initial_guess,
        target,
        tolerance=1e-3,
        maximum_iterations=100
    )

    result = result[0]
    print(xAx)

    actual = tf.linalg.matmul(target[0], tf.linalg.inv(matrix))

    print("Result: {}".format(result))
    print("Actual: {}".format(actual))
    print("Error: {}".format(tf.reduce_sum(
        tf.losses.mean_squared_error(actual, result))))
