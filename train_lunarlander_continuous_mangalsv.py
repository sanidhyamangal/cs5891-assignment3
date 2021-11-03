"""
author:Sanidhya Mangal
github:sanidhyamangal
"""
import tensorflow as tf # for deep learning
import optparse

tf.random.set_seed(0)
from ddpg_mangalsv import DDPG


if __name__ == "__main__":
    parser = optparse.OptionParser(
        "Script to train the MountainCar Continous model using Reinforce and DDPG algorithm"
    )

    parser.add_option(
        "-a",
        "--agent",
        dest="agent",
        help=
        "Select an agent to train MountainCarContinous model, default: ddpg",
        choices=["ddpg"],
        default="ddpg")
    parser.add_option(
        "-e",
        "--total_episodes",
        dest="total_episodes",
        help="Mention total number of epochs to run, default: 1000",
        type=int,
        default=1000)
    parser.add_option(
        "-l",
        "--lr",
        dest="lr",
        help="Specify the learning rate for training the model, default: 5e-4",
        type=float,
        default=5e-4)
    parser.add_option(
        "--num_hidden_states",
        help="Hidden layer architecture for the models, default: 512,512",
        default="512,512")
    parser.add_option(
        "--std",
        help="Specify the standard dev for the ou noise, default: 1e-1",
        type=float,
        default=1e-1)
    parser.add_option(
        "-b",
        "--buffer_size",
        help=
        "Specify the buffer size for keeping the length of replay buffer, default: 100000",
        type=int,
        default=100000)
    parser.add_option(
        "--batch_size",
        help="Specify the batch size for the training batch, default to 64",
        type=int,
        default=64)
    parser.add_option(
        "--plot_name",
        help="Specify the name of plot to save, default: lunarlander_mangalsv.png",
        type="str",
        default="lunarlander_mangalsv.png")
    (options, args) = parser.parse_args()

    n_hidden_states = list(
        map(lambda x: int(x), options.num_hidden_states.split(",")))

    if options.agent == "ddpg":
        ddpg = DDPG(problem="LunarLanderContinuous-v2",std=options.std,
                    num_hidden_states=n_hidden_states,
                    buffer_size=options.buffer_size,
                    batch_size=options.batch_size,
                    actor_lr=options.lr,
                    critic_lr=options.lr)

        ddpg.train(options.total_episodes,stopping_condition=200,plot_name=options.plot_name)
