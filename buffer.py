"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import numpy as np  # for matrix multiplication ops
import tensorflow as tf  # for deep learning
from models import Actor, Critic


# Class for implementing the buffer for the DDPG
class ExperienceBuffer(tf.Module):
    """
    Buffer Module for the DDPG algorithm
    """
    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 target_actor: Actor,
                 target_critic: Critic,
                 actor: Actor,
                 critic: Critic,
                 actor_optimizer: tf.optimizers.Optimizer,
                 critic_optimizer: tf.optimizers.Optimizer,
                 buffer_size: int = 1e5,
                 batch_size: int = 64,
                 name="ExperienceBuffer"):
        super().__init__(name=name)

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # To record how many times a buffer was called
        self.buffer_count = 0

        # create instance of all the models
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor = actor
        self.critic = critic

        # init optimizers
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

        # instead of a complete tuple arch we are maintaining a seperate np list for all the buffer sizes
        # it will provide us with more finer control
        self.state_buffer = np.zeros((self.buffer_size, num_states))
        self.action_buffer = np.zeros((self.buffer_size, num_actions))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, num_states))

    def record(self, observation_tuple):
        _idx = self.buffer_count % self.buffer_size

        self.state_buffer[_idx] = observation_tuple[0]
        self.action_buffer[_idx] = observation_tuple[1]
        self.reward_buffer[_idx] = observation_tuple[2]
        self.next_state_buffer[_idx] = observation_tuple[3]

        self.buffer_count += 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               gamma):
        # Train and update actor-critic networks
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True)
            critic_value = self.critic([state_batch, action_batch],
                                       training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss,
                                    self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        # work on grads for the actor
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_val = self.critic([state_batch, actions], training=True)

            # we use negative value for the critic_loss since we aim to maximize it
            actor_loss = -tf.math.reduce_mean(critic_val)

        actor_gradients = tape.gradient(actor_loss,
                                        self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables))

    def learn(self, gamma: int = 0.9):
        # Get sampling range
        record_range = min(self.buffer_count, self.buffer_size)

        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch,
                    gamma)
