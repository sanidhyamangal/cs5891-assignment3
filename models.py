"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf  # for deep learning
from tensorflow.keras import layers


def Actor(n_states: int, upper_bound: float, n_hidden_states: int = 256):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(n_states, ))
    out = layers.Dense(n_hidden_states, activation="relu")(inputs)
    out = layers.Dense(n_hidden_states, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh",
                           kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def Critic(n_states: int, n_action: int, n_hidden_states: int = 256):
    state_input = layers.Input(shape=(n_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(n_action))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(n_hidden_states, activation="relu")(concat)
    out = layers.Dense(n_hidden_states, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
