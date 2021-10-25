"""
author:Sanidhya Mangal
github:sanidhyamangal

The entire implementation for this model is heavily dependent on the following articles:
https://keras.io/examples/rl/ddpg_pendulum/
https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
"""


from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from buffer import ExperienceBuffer
from models import Actor, Critic
from utils import OUNoise


@tf.function
def update_target(target_weights, weights, tau):
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG:
    def __init__(self,
                 problem: str = "MountainCarContinuous-v0",
                 std: float = 0.2,
                 num_hidden_states: int = 128,
                 critic_lr: int = 0.002,
                 actor_lr: int = 0.001) -> None:
        self.env = gym.make(problem)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        self.ou_noise = OUNoise(mu=np.zeros(1), std=float(std) * np.ones(1))

        self.actor_model = Actor(upper_bound=self.upper_bound,
                                 hidden_state=num_hidden_states)
        self.critic_model = Critic(action_space=self.num_states,
                                   hidden_state=num_hidden_states)
        self.target_actor = Actor(upper_bound=self.upper_bound,
                                  hidden_state=num_hidden_states)
        self.target_critic = Critic(action_space=self.num_states,
                                    hidden_state=num_hidden_states)

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        self.buffer = ExperienceBuffer(self.num_states, self.num_actions,
                                       self.target_actor, self.target_critic,
                                       self.actor_model, self.critic_model,
                                       self.actor_optimizer,
                                       self.critic_optimizer, 50000, 256)

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound,
                               self.upper_bound)

        return [np.squeeze(legal_action)]

    def train(self,
              total_episodes: int,
              gamma: int = 0.99,
              tau: int = 5e-3,
              plot_name: Optional[str] = None):

    
        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []

        # Takes about 4 min to train
        for ep in range(total_episodes):

            prev_state = self.env.reset()
            episodic_reward = 0

            while True:
                # Uncomment this to see the Actor in action
                # But not in a python notebook.
                # env.render()

                tf_prev_state = tf.expand_dims(
                    tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state, self.ou_noise)
                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)

                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                self.buffer.learn(gamma)
                update_target(self.target_actor.variables,
                              self.actor_model.variables, tau)
                update_target(self.target_critic.variables,
                              self.critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-100:])
            print("Episode {}, Avg Reward is : {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)

        if plot_name is not None:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.savefig(plot_name)

ddpg = DDPG()

ddpg.train(5,plot_name="sample_plot.png")

