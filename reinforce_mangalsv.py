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
                 num_hidden_states: List[int] = [64,64],
                 lr: int = 5e-4) -> None:

        self.namespace = problem.split("-")[0]
        self.env = gym.make(problem)
        self.pg = PolicyGradient(self.env, num_hidden_states, std, lr)

    def train(self,
              total_episodes: int,
              start_episode:int,
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
                if episode < start_episode: action = self.env.action_space.sample()
                else: action = self.pg.get_action(state)
                # action, log_prob = policy_net.get_action(state)
                
                actions.append(action)
                new_state, reward, done, _ = self.env.step(action)
                rewards.append(reward)

                if done:
                    self.pg.train(state, action, rewards)
                    all_rewards.append(np.sum(rewards))
                    avg_rewards.append(np.mean(all_rewards[-100:]))
                    logger.info(f"Episode: {episode}, Average Reward: {avg_rewards[-1]}, Episodic Reward: {all_rewards[-1]}")
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


# env = gym.make('MountainCarContinuous-v0')

# max_episode_num = 225
# max_steps = 10000
# all_rewards = []
# avg_rewards = []

# for episode in range(max_episode_num):
#     state = env.reset()
#     actions = []
#     rewards = []

#     for steps in range(max_steps):
#         if episode < 200: action = env.action_space.sample()
#         else: action = policy_gradient.get_action(state)
#         # action, log_prob = policy_net.get_action(state)
        
#         actions.append(action)
#         new_state, reward, done, _ = env.step(action)
#         rewards.append(reward)

#         if done:
#             policy_gradient.train(state, action, rewards)
#             all_rewards.append(np.sum(rewards))
#             avg_rewards.append(np.mean(all_rewards[-100:]))
#             logger.info(f"Episode: {episode}, Average Reward: {avg_rewards[-1]}, Episodic Reward: {all_rewards[-1]}")
#             break

#         state = new_state

# plt.plot(all_rewards)
# plt.plot(all_rewards)
# plt.xlabel('Episode')
# plt.show()


# for episode in range(10):
#     state = env.reset()
#     log_probs = []
#     rewards = []

#     for steps in range(max_steps):
#         action= policy_gradient.get_action(state)
#         # action, log_prob = policy_net.get_action(state)
#         new_state, reward, done, _ = env.step(action)
#         rewards.append(reward)
#         env.render()

#         if done:
#             logger.info(f"Episode: {episode}, Episodic Reward: {np.sum(rewards)}")
#             # update_policy(policy_net, rewards, log_probs, optimizer)
#             # numsteps.append(steps)
#             # avg_numsteps.append(np.mean(numsteps[-10:]))
#             # all_rewards.append(np.mean(rewards[-100:]))
#             # logger.info(f"Episode: {episode}, Average Reward: {all_rewards[-1]}")
#             break

#         state = new_state

reinforce = Reinforce(std=0.5)

reinforce.train(300,5,plot_name="reinforce_mc.png")