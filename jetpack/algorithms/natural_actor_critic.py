"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.algorithms.actor_critic import ActorCritic
from jetpack.algorithms.natural_policy_gradient import NaturalPolicyGradient


class NaturalActorCritic(ActorCritic):

    def __init__(
        self,
        policy,
        critic,
        gamma=1.0,
        actor_delay=1,
        monitor=None,
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            gamma=gamma,
            actor_delay=actor_delay,
            monitor=monitor,
        )

    def update_policy(
        self,
        observations,
        actions,
        returns
    ):
        NaturalPolicyGradient.update_policy(
            self,
            observations,
            actions,
            returns
        )


