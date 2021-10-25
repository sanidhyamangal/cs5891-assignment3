"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import Any
import numpy as np  # for matrix multiplication


class OUNoise:
    """
    Noise class to implement add noise to the action selection space for generalizing
    the exploration phase so that agent can generalize well.
    Details could be found here: https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
    """
    def __init__(self,
                 mu: float,
                 std: float,
                 theta: float = 0.15,
                 dt: float = 1e-2,
                 x_init=None) -> None:

        # init the hyper params
        self.mu = mu
        self.std = std
        self.theta = theta
        self.dt = dt
        self.x_init = x_init

        # perform initial reset for the noise to init x_prev based on x_init
        self.reset()

    # overwrite call val for noise ops
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # compute the noisy pertubrations using formulae given in
        # https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

        noisy_pertubrations = (
            self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
            self.std * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))

        # assign these new noise pertubrations to prev step
        self.x_prev = noisy_pertubrations

        # return X_{n+1} noise pertubrations
        return noisy_pertubrations

    # define reset function to reinit x_prev value
    def reset(self) -> None:
        if self.x_init is not None: self.x_prev = self.x_init
        else: self.x_prev = np.zeros_like(self.mu)
