"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import tensorflow as tf # for deep learning
import numpy as np # for matrix multiplication

from tensorflow.keras import models, layers

class PolicyNetwork(models.Model):

    def __init__(self, env_space:int, action_space:int, hidden_state:int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.env_space = env_space
        self.action_space = action_space

        self.model = models.Sequential([
            layers.Input(shape=env_space),
            layers.Dense(units=hidden_state, activation="relu"),
            layers.Dense(units=action_space, activation=tf.nn.softmax)
        ])

    def call(self, state, training=None):
        return self.model(state)
    

    @tf.function
    def get_action(self, state:np.array):
        state_tensor = tf.convert_to_tensor(state[:, np.newaxis], dtype=tf.float32)
        prob = self.call(state_tensor)

        best_action = np.random.choice(self.action_space, p=tf.squeeze(prob).numpy())

        log_probab = tf.math.log(tf.squeeze(prob)[best_action])

        return best_action, log_probab