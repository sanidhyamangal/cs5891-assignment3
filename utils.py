"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

from typing import Any
import numpy as np # for matrix multiplication

class Noise:
    def __init__(self,mu:float, std:float, theta:float=0.15, dt:float=1e-2 ,x_init=None) -> None:
        self.mu = mu
        self.std = std
        self.theta = theta
        self.dt = dt
        self.x_init = x_init

        self.reset()

    # overwrite call val for noise ops
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # compute the noisy pertubrations using formulae given in https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
        noisy_pertubrations = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        # assign these new noise pertubrations to prev step
        self.x_prev = noisy_pertubrations

        # return X_{n+1} noise pertubrations
        return noisy_pertubrations

    # define reset function to reinit x_prev value
    def reset(self) -> None:
        if self.x_init is not None:self.x_prev = self.x_init
        else: self.x_prev = np.zeros_like(self.mu)