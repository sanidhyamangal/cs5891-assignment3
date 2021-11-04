"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import List

import numpy as np  # for matrix multiplication
import tensorflow as tf  # for deep learning
from tensorflow.keras import layers, models


def PolicyNetwork(n_states: int,
                  n_actions: int,
                  upper_bound: int,
                  n_hidden_states: List[int] = [64, 64]):
    """
    Function to construct policy network for reinforce
    """

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    # define input layer for the shape of input
    inputs = layers.Input(shape=(n_states, ))

    # define linear layer with relu activations using hidden layer unit params
    out = layers.Dense(n_hidden_states[0], activation="relu")(inputs)

    for units in n_hidden_states[1:]:
        out = layers.Dense(units, activation="relu")(out)
    
    # mu aka mean for the normal distribution
    output_mu = layers.Dense(n_actions,
                             activation="tanh",
                             kernel_initializer=last_init)(out)

    # output of mu for the upper bound
    mu_output = layers.Lambda(lambda x: x * upper_bound)(output_mu)

    # std dev for distribution
    std_output = layers.Dense(n_actions, activation='softplus')(out)

    # defined tf model which takes state as input and returns mu and std as output
    model = tf.keras.Model(inputs, [mu_output, std_output])
    return model


def Actor(n_states: int,
          n_actions: int,
          upper_bound: float,
          n_hidden_states: List[int] = [64, 64]):
    
    """
    Function to construct actor nn for DDPG
    """

    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(n_states, ))
    out = layers.Dense(n_hidden_states[0], activation="relu")(inputs)

    for units in n_hidden_states[1:]:
        out = layers.Dense(units, activation="relu")(out)

    # Return output as tanh activation for damppening the output from -1 to 1
    outputs = layers.Dense(n_actions,
                           activation="tanh",
                           kernel_initializer=last_init)(out)


    # multiply the output with the upper bound
    outputs = outputs * upper_bound

    # construct model with the inputs and outputs
    model = tf.keras.Model(inputs, outputs)
    return model


def Critic(n_states: int,
           n_action: int,
           n_hidden_states: List[int] = [64, 64]):
    
    """
    Function to construct critic nn for DDPG
    """

    # create an encoder for state to encode them into 32 dense units
    state_input = layers.Input(shape=(n_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input and embeed them into 32 units
    action_input = layers.Input(shape=(n_action))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    # hidden layers for the network
    out = layers.Dense(n_hidden_states[0], activation="relu")(concat)

    for units in n_hidden_states[1:]:
        out = layers.Dense(units, activation="relu")(out)

    # generate output stating how good the policy is
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
