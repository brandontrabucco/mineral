"""Author: Brandon Trabucco, Copyright 2019"""


import io
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_to_tensor(
    xs,
    ys,
    title,
    x_label,
    y_label
):
    plt.clf()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return tf.expand_dims(
        tf.image.decode_png(buffer.getvalue(), channels=4),
        0
    )
