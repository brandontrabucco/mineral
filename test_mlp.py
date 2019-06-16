"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.fully_connected import FullyConnectedPolicy, FullyConnectedQF


if __name__ == "__main__":


    policy = FullyConnectedPolicy(
        [6, 6, 1]
    )

    qf = FullyConnectedQF(
        [6, 6]
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