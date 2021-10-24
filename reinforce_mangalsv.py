"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import gym  # for open ai gym
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for matrix maths
import tensorflow as tf  # for deep learning

from models import PolicyNetwork

class ReinforceAgent:
    def __init__(self, state_space:int ,action_space:int, ) -> None:
        pass

# def update_policy(model: PolicyNetwork,
#                   rewards,
#                   log_probs,
#                   optimizer: tf.optimizers.Optimizer,
#                   gamma=0.9):
#     _discounted_rewards = []

#     with tf.GradientTape() as tape:

#         for t in range(len(rewards)):
#             G_t, pw = 0, 0
#             for r in rewards[t:]:
#                 G_t += gamma**pw * r
#                 pw += 1

#             _discounted_rewards.append(G_t)

#         _discounted_rewards = np.array(_discounted_rewards)
#         _discounted_rewards = (
#             _discounted_rewards - _discounted_rewards.mean()) / (
#                 _discounted_rewards.std() + 1e-9)  # normalize discounted rewards

#         policy_gradients = [
#             -log_prob * Gt for log_prob, Gt in zip(log_probs, _discounted_rewards)
#         ]
    
#     import pdb;pdb.set_trace()
#     gradient=tape.gradient(tf.reduce_sum(policy_gradients), model.trainable_variables)
#     optimizer.apply_gradients(zip(gradient, model.trainable_variables))


# env = gym.make('LunarLander-v2')
# policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n,
#                            128)

# max_episode_num = 5000
# max_steps = 10000
# numsteps = []
# avg_numsteps = []
# all_rewards = []

# optimizer = tf.optimizers.Adam
# for episode in range(max_episode_num):
#     state = env.reset()
#     log_probs = []
#     rewards = []

#     for steps in range(max_steps):
#         action, log_prob = policy_net.get_action(state)
#         new_state, reward, done, _ = env.step(action)
#         log_probs.append(log_prob)
#         rewards.append(reward)

#         if done:
#             update_policy(policy_net, rewards, log_probs, optimizer)
#             numsteps.append(steps)
#             avg_numsteps.append(np.mean(numsteps[-10:]))
#             all_rewards.append(np.sum(rewards))
#             if episode % 100 == 0:
#                 print(
#                     "episode: {}, total reward: {}, average_reward: {}, length: {}\n"
#                     .format(episode, np.round(np.sum(rewards), decimals=3),
#                             np.round(np.mean(all_rewards[-10:]), decimals=3),
#                             steps))
#             break

#         state = new_state

# plt.plot(numsteps)
# plt.plot(avg_numsteps)
# plt.xlabel('Episode')
# plt.show()
