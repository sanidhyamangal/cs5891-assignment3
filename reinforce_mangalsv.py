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

        # define the hyper params for the reinforce
        self.namespace = problem.split("-")[0]
        self.env = gym.make(problem)

        # constricy the policy gradient instance for training
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

        # iter for each episode in range
        for episode in range(total_episodes):
            state = self.env.reset()
            rewards = []

            # is start episode comes work on sampling the actions from the state
            if episode == start_episode:
                logger.info("Exploration-Exploitation Begin !!")

            while True:
                # for warm start of the network to explore the env
                if episode < start_episode:
                    action = self.env.action_space.sample()
                else:
                    # sample the action from the policy network
                    action = self.pg.get_action(state)

                # take the action in the env
                new_state, reward, done, _ = self.env.step(action)

                # append the rewards and store it for updating the policy
                rewards.append(reward)

                if done:
                    # if done train the network based on actions rewards
                    self.pg.train(state, action, rewards, gamma)

                    # append the rewards for plotting
                    all_rewards.append(np.sum(rewards))
                    # take a mean of last 100 rewards for plots and stopping condition
                    avg_rewards.append(np.mean(all_rewards[-100:]))

                    # log the progress of the episode
                    logger.info(
                        f"Episode: {episode}, Average Reward: {avg_rewards[-1]}, Episodic Reward: {all_rewards[-1]}"
                    )
                    break

                state = new_state

            # if last avg reward is greater than stopping condition then coin problem as solved
            if avg_rewards[-1] >= stopping_condition:
                logger.info("Problem Solved at : {}".format(episode))
                break

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
        # save weights for the reinforce
        if not os.path.exists("trained_models"): os.mkdir("trained_models")

        self.pg.model.save_weights(
            f"trained_models/{self.namespace}-reinforce_mangalsv.h5")


def test_reinforce(problem: str = "MountainCarContinuous-v0",
                   episodes: int = 10,
                   num_hidden_states: List[int] = [64, 64],
                   weight_path: Optional[str] = None) -> None:

    """
    Function to test the reinforce algorithm
    """
    # define hyperparams
    namespace = problem.split("-")[0]
    env = gym.make(problem) # define the gym env
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]

    # construct the policy network
    model = PolicyNetwork(num_states, num_actions, upper_bound,
                          num_hidden_states)

    # define action method to sample actions for the given state from the model
    def get_action(state):
        state = np.reshape(state, [1, num_states])
        mu, std = model(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=num_actions)

    # load the weights for the policy network model
    if not weight_path:
        model.load_weights(f"trained_models/{namespace}-reinforce_mangalsv.h5")
    else:
        model.load_weights(weight_path)

    # iterate through each episode and render the enviornment
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
