"""
author: Sanidhya Mangal
github: sanidhyamangal
"""

import numpy as np  # for matrix multiplication
import tensorflow as tf  # for deep learning

from models import PolicyNetwork  # import policy network


class PolicyGradient:
    def __init__(self, env, n_hidden_states, std_bound, lr):

        # define hyper params for the policy gradient op
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = std_bound

        # create a policy network for value function approximation
        self.model = PolicyNetwork(self.state_dim,
                                   self.action_dim,
                                   upper_bound=self.action_bound,
                                   n_hidden_states=n_hidden_states)
        
        # define the optimizer for updating the weight
        self.opt = tf.keras.optimizers.Adam(lr)

    # function to get action for the given state
    def get_action(self, state):
        # reshape the action for feedingn it to policy network
        state = np.reshape(state, [1, self.state_dim])

        # get the mean and std dev from the model as output
        mu, std = self.model(state)

        # reduce the dim of mean and std for sampling the action from the gaussian normal distribution
        mu, std = mu[0], std[0]

        # return the action along with the random noise for the learning
        return np.random.normal(
            mu, std, size=self.action_dim) # work on removing it.
    

    # compute the log pdf for computing the loss for the
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, -self.std_bound, self.std_bound)
        var = std**2 # compute variation using std

        # logpdf = -0.5*(x-mu)^2 / (sigma^2 - 0.5log(2*pi*sigma^2))
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        
        # sum the log probs
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, rewards, gamma: int = 0.99):
        # compute discounted rewards for loss function
        discounted_rewards = rewards * (gamma**np.arange(len(rewards)))

        # compute the log pdf for the given action
        log_policy_pdf = self.log_pdf(mu, std, actions)

        # multiply log prob with the discounted rewards and return their sum.
        loss_policy = log_policy_pdf * discounted_rewards
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, rewards, gamma):
        # perform training ops for the gradient decent
        with tf.GradientTape() as tape:
            # run a sample on the state
            mu, std = self.model(states, training=True)

            # compute the loss
            loss = self.compute_loss(mu, std, actions, rewards, gamma)
        
        # find the gradients for the log of pdf
        grads = tape.gradient(loss, self.model.trainable_variables)

        # update the optimizer by applying gradients on the policy
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
