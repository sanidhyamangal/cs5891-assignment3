"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import os  # for os related op
from typing import List, Optional

import gym  # for open ai gym
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for matrix maths
import tensorflow as tf  # for deep learning

from logger import logger
from models import PolicyNetwork  # for logging the through puts
from policy_gradient import PolicyGradient

tf.random.set_seed(0)
np.random.seed(0)


class Reinforce:
    def __init__(self,
                 problem: str = "MountainCarContinuous-v0",
                 std: float = 0.2,
                 num_hidden_states: List[int] = [64, 64],
                 lr: int = 5e-4) -> None:

        self.namespace = problem.split("-")[0]
        self.env = gym.make(problem)
        self.pg = PolicyGradient(self.env, num_hidden_states, std, lr)

    def train(self,
              total_episodes: int,
              start_episode: int,
              gamma: int = 0.99,
              stopping_condition: float = 90.0,
              plot_name: Optional[str] = None):

        logger.info("Resetting Epsidode Reward and avg reward list")
        # To store reward history of each episode
        all_rewards = []
        # To store average reward history of last few episodes
        avg_rewards = []

        for episode in range(total_episodes):
            state = self.env.reset()
            actions = []
            rewards = []

            if episode == start_episode:
                logger.info("Exploration-Exploitation Begin !!")

            while True:
                if episode < start_episode:
                    action = self.env.action_space.sample()
                else:
                    action = self.pg.get_action(state)
                # action, log_prob = policy_net.get_action(state)

                actions.append(action)
                new_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)

                if done:
                    self.pg.train(state, action, rewards, gamma)
                    all_rewards.append(np.sum(rewards))
                    avg_rewards.append(np.mean(all_rewards[-100:]))
                    logger.info(
                        f"Episode: {episode}, Average Reward: {avg_rewards[-1]}, Episodic Reward: {all_rewards[-1]}"
                    )
                    break

                state = new_state

        # call save weights method
        self.save_weights()
        if plot_name is not None:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_rewards)
            plt.plot(all_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.savefig(plot_name)
            plt.clf()

        logger.info("Training Finished !!")

    def save_weights(self) -> None:

        if not os.path.exists("trained_models"): os.mkdir("trained_models")

        self.pg.model.save_weights(
            f"trained_models/{self.namespace}-reinforce_mangalsv.h5")


def test_reinforce(problem: str = "MountainCarContinuous-v0",
                   episodes: int = 10,
                   num_hidden_states: List[int] = [64, 64],
                   weight_path: Optional[str] = None) -> None:

    namespace = problem.split("-")[0]
    env = gym.make(problem)
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]

    model = PolicyNetwork(num_states, num_actions, upper_bound,
                          num_hidden_states)

    def get_action(state):
        state = np.reshape(state, [1, num_states])
        mu, std = model(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=num_actions)

    if not weight_path:
        model.load_weights(f"trained_models/{namespace}-reinforce_mangalsv.h5")
    else:
        model.load_weights(weight_path)

    for _ in range(episodes):
        state = env.reset()
        episodic_reward = 0

        while True:
            env.render()

            action = get_action(state)

            if num_actions > 1:
                action = action[0]

            # Recieve state and reward from environment.
            next_state, reward, done, info = env.step(action)

            episodic_reward += reward

            if done:
                break

            state = next_state

        logger.info("Episodic Reward: {}".format(episodic_reward))
