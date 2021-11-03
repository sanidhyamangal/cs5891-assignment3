import gym
import numpy as np  # for matrix multiplication
import tensorflow as tf  # for deep learning

from models import PolicyNetwork


class PolicyGradient:
    def __init__(self,env,n_hidden_states, std_bound, lr):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = std_bound
        self.model = PolicyNetwork(self.state_dim, self.action_dim, upper_bound=self.action_bound,n_hidden_states=n_hidden_states)
        self.opt = tf.keras.optimizers.Adam(lr)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim) + np.random.normal(0,0.5)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, -self.std_bound, self.std_bound)
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, rewards, gamma:int=0.99):
        discounted_rewards = rewards * gamma ** np.arange(len(rewards))
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * discounted_rewards
        return tf.reduce_sum(-loss_policy)

    # @tf.function
    def train(self, states, actions, rewards):
        with tf.GradientTape() as tape:
            mu, std = self.model(np.resize(states,[1,self.state_dim]), training=True)
            loss = self.compute_loss(mu, std, actions, rewards)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss