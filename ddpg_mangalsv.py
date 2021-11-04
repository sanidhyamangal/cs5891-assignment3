"""
author:Sanidhya Mangal
github:sanidhyamangal

The entire implementation for this model is heavily dependent on the following articles:
https://keras.io/examples/rl/ddpg_pendulum/
https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
"""

import os # for os related ops
from typing import List, Optional # for defining the typings of the args and return type

import gym # for open ai env
import matplotlib.pyplot as plt # for plotting the envs
import numpy as np # for matrix multiplication
import tensorflow as tf

from buffer import ExperienceBuffer # for storing the experience as a buffer for DDPG
from models import Actor, Critic
from utils import OUNoise # noise term for DDPG algorithm
from logger import logger # logger for logging the events for the run and test set


@tf.function
def update_target(target_weights, weights, tau) -> None:
    """Perform soft update for the target policies in ddpg"""
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


# DDPG agent def.
class DDPG:
    def __init__(self,
                 problem: str = "MountainCarContinuous-v0",
                 std: float = 0.2,
                 num_hidden_states: int = 64,
                 buffer_size: int = 100000,
                 batch_size=128,
                 critic_lr: int = 0.0005,
                 actor_lr: int = 0.0005) -> None:

        # init all the hyperparms and model defination
        self.namespace = problem.split("-")[0]
        self.env = gym.make(problem)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # init models and noise object
        self.ou_noise = OUNoise(mu=np.zeros(1), std=float(std) * np.ones(1))

        self.actor_model = Actor(n_states=self.num_states,
                                 n_actions=self.num_actions,
                                 upper_bound=self.upper_bound,
                                 n_hidden_states=num_hidden_states)
        self.critic_model = Critic(n_states=self.num_states,
                                   n_action=self.num_actions,
                                   n_hidden_states=num_hidden_states)
        self.target_actor = Actor(n_states=self.num_states,
                                  n_actions=self.num_actions,
                                  upper_bound=self.upper_bound,
                                  n_hidden_states=num_hidden_states)
        self.target_critic = Critic(n_states=self.num_states,
                                    n_action=self.num_actions,
                                    n_hidden_states=num_hidden_states)
        
        # create adam optimizers for optimizing the models
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        # set same initial weights for the critic and target
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.target_actor.set_weights(self.actor_model.get_weights())

        # init a experience buffer for storing all the record runs for the algorithm
        self.buffer = ExperienceBuffer(self.num_states, self.num_actions,
                                       self.target_actor, self.target_critic,
                                       self.actor_model, self.critic_model,
                                       self.actor_optimizer,
                                       self.critic_optimizer, buffer_size,
                                       batch_size)
    

    def policy(self, state, noise_object) -> List[np.array]:
        """Define a deterministic policy for the DDPG algorithm, 
        returns an action in form of np array"""

        # reterieve the actions from the actor model
        sampled_actions = tf.squeeze(self.actor_model(state))

        # generate a noise term
        noise = noise_object()

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds by cliping its value in lower bound and upper bound
        legal_action = np.clip(sampled_actions, self.lower_bound,
                               self.upper_bound)

        # retuern the action as an np array
        return [np.squeeze(legal_action)]

    def train(self,
              total_episodes: int,
              gamma: int = 0.99,
              tau: int = 5e-3,
              stopping_condition: float = 90.0,
              plot_name: Optional[str] = None):

        logger.info("Resetting Epsidode Reward and avg reward list")
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

                # convert the action to feed into the actor network to sample action
                tf_prev_state = tf.expand_dims(
                    tf.convert_to_tensor(prev_state), 0)

                # retrieve the action for the given state value
                action = self.policy(tf_prev_state, self.ou_noise)
                # Recieve state and reward from environment.

                # if the action dim is greater than 1 then squeeze the action space
                if self.num_actions > 1:
                    action = action[0]

                # get the next state, reward, and status of the env based on sampled action
                state, reward, done, info = self.env.step(action)
                
                # record the observation into the experience buffer to bootstrapping
                self.buffer.record((prev_state, action, reward, state))

                # update the episodic reward
                episodic_reward += reward

                # update the policies
                self.buffer.learn(gamma)

                # perform soft updates on target actor and critic
                update_target(self.target_actor.variables,
                              self.actor_model.variables, tau)
                update_target(self.target_critic.variables,
                              self.critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break
                
                # update the prev state with the next state to sample new action
                prev_state = state

            # append the episodic rewards for checking the avg reward and plotting
            ep_reward_list.append(episodic_reward)

            # Mean of last 100 episodes
            avg_reward = np.mean(ep_reward_list[-100:])
            # print the logs for the episode
            logger.info(
                "Episode {}, Avg Reward is : {}, Episodic Reward: {}".format(
                    ep, avg_reward, episodic_reward))

            # append the average reward into the list for plotting
            avg_reward_list.append(avg_reward)

            # if last avg reward is greater than stopping condition then coin problem as solved
            if avg_reward_list[-1] >= stopping_condition:
                logger.info("Problem Solved at : {}".format(ep))
                break

        # call save weights method
        self.save_weights()
        
        # plot the avg rewards and episodic rewards for entire training lifecycle.
        if plot_name is not None:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.plot(ep_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.savefig(plot_name)
            plt.clf()

        logger.info("Training Finished !!")

    def save_weights(self) -> None:
        """Function to save the weights for the models"""

        if not os.path.exists("trained_models"): os.mkdir("trained_models")

        self.actor_model.save_weights(
            f"trained_models/{self.namespace}-actor_mangalsv.h5")
        self.critic_model.save_weights(
            f"trained_models/{self.namespace}-critic_mangalsv.h5")
        self.target_actor.save_weights(
            f"trained_models/{self.namespace}-tg_actor_mangalsv.h5")
        self.target_critic.save_weights(
            f"trained_models/{self.namespace}-tg_critic_mangalsv.h5")


def test_ddpg(problem: str = "MountainCarContinuous-v0",
              episodes: int = 3,
              num_hidden_states: int = 64,
              weight_path: Optional[str] = None) -> None:
    """
    Method to perform test using ddpg algorithm
    """

    # extract the hyperparams based on the env and hidden units
    namespace = problem.split("-")[0]
    env = gym.make(problem) # create an open ai env
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    # create an actor model for formulating the actions
    actor_model = Actor(n_states=num_states,
                        n_actions=num_actions,
                        upper_bound=upper_bound,
                        n_hidden_states=num_hidden_states)

    # load the weights for the actor model
    if not weight_path:
        actor_model.load_weights(
            f"trained_models/{namespace}-actor_mangalsv.h5")
    else:
        actor_model.load_weights(weight_path)

    # define a policy function to work with deep deterministic policy
    def policy(state):
        # *Same as policy function in DDPG agent class*
        sampled_actions = tf.squeeze(actor_model(state))

        sampled_actions = sampled_actions.numpy()

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return [np.squeeze(legal_action)]

    # iterate through each episode to render the episodic tasks
    for _ in range(episodes):
        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            env.render()

            # sample the action based on policy
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state)
            
            # perform action dim compatiblity 
            if num_actions > 1:
                action = action[0]

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            # update the episodic task
            episodic_reward += reward

            # break if done
            if done:
                break
            
            # update the prev state to next state
            prev_state = state

        # log the episodic rewards
        logger.info("Episodic Reward: {}".format(episodic_reward))
