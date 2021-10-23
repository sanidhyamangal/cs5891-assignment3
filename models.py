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
            layers.Linear(units=hidden_state, activation="relu"),
            layers.Dense(units=action_space, activation=tf.nn.softmax)
        ])

    def call(self, state, training=None):
        return self.model(state)
    

    def get_action(self, state:np.array):
        prob = self.call(state[np.newaxis, :])

        best_action = np.random.choice(self.action_space, p=tf.squeeze(prob).numpy())

        log_probab = tf.math.log(tf.squeeze(prob)[best_action])

        return best_action, log_probab
    
    # # @tf.function
    def update_policy(self, rewards, log_probs, optimizer:tf.optimizers.Optimizer ,gamma=0.9):
        _discounted_rewards = []
        
        for t in range(len(rewards)):
            G_t, pw = 0, 0
            for r in rewards[t:]:
                G_t += gamma**pw * r
                pw += 1
            
            _discounted_rewards.append(G_t)


        _discounted_rewards = np.array(_discounted_rewards)
        _discounted_rewards = (_discounted_rewards - _discounted_rewards.mean()) / (_discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradients = [-log_prob*Gt for log_prob, Gt in zip(log_probs, _discounted_rewards)]

        optimizer.apply_gradients(zip(policy_gradients, self.weights))