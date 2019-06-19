"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_policy import DensePolicy
from jetpack.networks.dense_qf import DenseQF
from jetpack.networks.dense_vf import DenseVF


if __name__ == "__main__":

    policy = DensePolicy(
        [6, 6, 1]
    )

    qf = DenseQF(
        [6, 6, 1]
    )

    vf = DenseVF(
        [6, 6, 1]
    )

    x = tf.random.normal([32, 6])
    y = tf.random.normal([32, 1])

    for i in range(100):

        with tf.GradientTape() as tape:
            policy_loss = tf.reduce_mean(tf.losses.mean_squared_error(y, policy(x)))
            policy.minimize(policy_loss, tape)

        print("policy_loss", policy_loss)

    for i in range(100):

        with tf.GradientTape() as tape:
            qf_loss = tf.reduce_mean(tf.losses.mean_squared_error(y, qf(x)))
            qf.minimize(qf_loss, tape)

        print("qf_loss", qf_loss)

    for i in range(100):

        with tf.GradientTape() as tape:
            vf_loss = tf.reduce_mean(tf.losses.mean_squared_error(y, vf(x)))
            qf.minimize(vf_loss, tape)

        print("vf_loss", vf_loss)