"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import optparse

from ddpg_mangalsv import test_ddpg
from reinforce_mangalsv import test_reinforce

if __name__ == "__main__":
    parser = optparse.OptionParser(
        "Script to test the MountainCar Continous model using Reinforce and DDPG algorithm"
    )

    parser.add_option(
        "-a",
        "--agent",
        dest="agent",
        help=
        "Select an agent to test MountainCarContinous model, default: ddpg",
        choices=["reinforce", "ddpg"],
        default="ddpg")
    parser.add_option(
        "-e",
        "--total_episodes",
        dest="total_episodes",
        help="Mention total number of epochs to run, default: 10",
        type=int,
        default=10)
    parser.add_option(
        "--num_hidden_states",
        help="Hidden layer architecture for the models, default: 64,64",
        default="64,64")
    parser.add_option("-w",
                      "--weight_path",
                      help="Specify the path of weights to load")
    (options, args) = parser.parse_args()

    n_hidden_states = list(
        map(lambda x: int(x), options.num_hidden_states.split(",")))

    if options.agent == "ddpg":
        test_ddpg(episodes=options.total_episodes,
                  num_hidden_states=n_hidden_states,
                  weight_path=options.weight_path)
    else:
        test_reinforce(episodes=options.total_episodes,
                       num_hidden_states=n_hidden_states,
                       weight_path=options.weight_path)
