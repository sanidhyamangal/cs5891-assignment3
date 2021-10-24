"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

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
