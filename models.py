"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf  # for deep learning
from tensorflow.keras import models, layers


class BaseLinearModel(models.Model):
    """
    Define a base linear model which would be later used by actor and critic methods
    """
    def __init__(self, action_space: int, hidden_state: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = models.Sequential([
            layers.Dense(hidden_state, activation="relu"),
            layers.Dense(hidden_state, activation="relu"),
            layers.Dense(action_space)
        ])


class Actor(BaseLinearModel):
    def __init__(self, upper_bound: int, hidden_state: int, *args, **kwargs):
        self.upper_bound = upper_bound
        super().__init__(1, hidden_state, *args, **kwargs)

    def call(self, state, training=None):
        x = self.model(state, training=training)
        x = tf.nn.tanh(x)

        return x * self.upper_bound


class Critic(BaseLinearModel):
    def __init__(self, action_space: int, hidden_state: int, *args, **kwargs):
        super().__init__(action_space, hidden_state, *args, **kwargs)

        self.state_embeds = models.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu")
        ])

        self.action_embeds = layers.Dense(32, activation="relu")

    def call(self, state, action, training=None):
        state = self.state_embeds(state)
        action = self.action_embeds(action)

        _input = tf.concat([state, action], axis=1)
        x = self.model(_input, training=training)

        return x
