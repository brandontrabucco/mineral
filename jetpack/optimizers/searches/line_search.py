"""Author: Brandon Trabucco, Copyright 2019"""


def line_search(
    loss_function,
    network,
    grad,
    alpha,
    scale_factor=0.5,
    iterations=100
):
    original_weights = network.get_weights()
    def wrapped_loss_function(
        beta
    ):
        network.set_weights([
            x - beta * dx
            for x, dx in zip(original_weights, grad)
        ])
        return loss_function()
    best_alpha = alpha
    best_loss = float("inf")
    for i in range(iterations):
        loss = wrapped_loss_function(alpha)
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
        if i < iterations - 1:
            alpha *= scale_factor
    network.set_weights(original_weights)
    return [best_alpha * dx for dx in grad]
