"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import gym  # for open ai gym
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for matrix maths
import tensorflow as tf  # for deep learning

from models import PolicyNetwork

def update_network(network, rewards, states, gamma:float=0.9):
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    # standardise the rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    states = np.vstack(states)
    loss = network.train_on_batch(states, discounted_rewards)
    return loss

def main():
    env = gym.make('LunarLander-v2')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    policy_net.compile(loss='categorical_crossentropy',optimizer=tf.optimizers.Adam())

    num_episodes = 10000000
    for episode in range(num_episodes):
        state = env.reset()
        rewards = []
        states = []
        actions = []
        while True:
            action = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            actions.append(action)

            if done:
                loss = update_network(policy_net, rewards, states)
                tot_reward = sum(rewards)
                print(f"Episode: {episode}, Reward: {tot_reward}, avg loss: {loss:.5f}")
                break

            state = new_state

main()