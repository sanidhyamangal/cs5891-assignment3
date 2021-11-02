import gym

from policy_gradient import PolicyGradient


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")
    model = PolicyGradient(env,num_iterations=50,action_variance=0.1,learning_rate=0.1,gamma=0.99,hidden1_size=128,hidden2_size=128)
    model.train()
    # model.make_video()