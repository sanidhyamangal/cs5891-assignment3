"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List

import numpy as np  # for matrix multiplication
import tensorflow as tf  # for deep learning
import tensorflow_probability as tfp  # for probablistic
from tensorflow.keras import layers, models


class PolicyNetwork(models.Model):

    def __init__(self, env_space:int, action_space:int, hidden_state:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env_space = env_space
        self.action_space = action_space

        self.model = models.Sequential([
            layers.Dense(units=hidden_state, activation="relu"),
            layers.Dense(units=action_space, activation="softmax")
        ])

    def call(self, state, training=None):
        return self.model(state)
    

    def get_action(self, state:np.array):
        # compute the softmax probablities for the state
        prob = self.call(np.array([state]))
        
        # compute the prob distributions for the given probs
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)

        # sample the probablity distribution
        action = dist.sample()

        # return action with highest probablity
        return int(action.numpy()[0])

def Actor(n_states: int,
          n_actions: int,
          upper_bound: float,
          n_hidden_states: List[int] = [64, 64]):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(n_states, ))
    out = layers.Dense(n_hidden_states[0], activation="relu")(inputs)

    for units in n_hidden_states[1:]:
        out = layers.Dense(units, activation="relu")(out)

    outputs = layers.Dense(n_actions,
                           activation="tanh",
                           kernel_initializer=last_init)(out)

    # [2,shape_action_Space]
    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def Critic(n_states: int,
           n_action: int,
           n_hidden_states: List[int] = [64, 64]):
    state_input = layers.Input(shape=(n_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(n_action))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(n_hidden_states[0], activation="relu")(concat)

    for units in n_hidden_states[1:]:
        out = layers.Dense(units, activation="relu")(out)

    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
