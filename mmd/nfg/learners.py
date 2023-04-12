"""Learners for normal-form games.
"""

from typing import Callable, Protocol

import numpy as np
from numba import njit
from open_spiel.python.algorithms.mmd_dilated import softmax

from ..utils import project


class Learner(Protocol):
    """Protocol class for learners"""

    def update(self, payoffs: np.ndarray) -> None:
        """Update internal state of learner using `payoffs`"""

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""

    def test_policy(self) -> np.ndarray:
        """Return policy for testing"""


class MMD(Learner):
    def __init__(
        self,
        n_actions: int,
        initializer: Callable[[int], np.ndarray],
        lr: Callable[[int], float],
        temp: Callable[[int], float],
        magnet_initializer: Callable[[int], np.ndarray],
        magnet_lr: Callable[[int], float],
        noise: Callable[[np.ndarray], np.ndarray],
        use_magnet: bool = True,
        test_magnet: bool = False,
        test_average=False,
    ):
        """Implements of magnetic mirror descent

        Args:
            n_actions: Number of actions for the learner
            initializer: Initializer for learner's policy
            lr: Stepsize schedule
            temp: Regularization temperature schedule
            magnet_initializer: Initializer for learner's magnet
            magnet_lr: Stpesize schedule for magnet

        Attributes:
            n_actions
            initializer
            lr
            temp
            magnet_initializer
            magnet_lr
            policy: Current policy
            iteration: Number of completed updates
        """
        self.initializer = initializer
        self.lr = lr
        self.temp_fn = temp
        self.magnet_initializer = magnet_initializer
        self.magnet = project(magnet_initializer(n_actions))
        self.magnet_lr = magnet_lr
        self.policy: np.ndarray = project(initializer(n_actions))
        self.noise_fn = noise
        self.cum_policy: np.ndarray = self.policy.copy()
        self.exp_avg_policy: np.ndarray = self.policy.copy()
        self.use_magnet = use_magnet
        self.test_magnet = test_magnet
        self.test_average = test_average
        self.iteration: int = 1
        self.grad = np.array([0, 0])
        self.temp = self.temp_fn(self.iteration)

    # @njit
    def update(self, qs: np.ndarray) -> None:
        """Update internal state of learner using `payoffs`"""
        qs = self.noise_fn(qs)
        self.grad = qs.copy()
        self.iteration += 1
        lr = self.lr(self.iteration)
        self.temp = self.temp_fn(self.iteration)
        mag_lr = self.magnet_lr(self.iteration)
        magnet_part = self.magnet ** (lr * self.temp) if self.use_magnet else 1
        policy = project(
            np.power(
                self.policy * np.exp(lr * qs) * magnet_part,
                1 / (1 + lr * self.temp),
            )
        )
        magnet = project(
            np.power(self.policy, mag_lr) * np.power(self.magnet, 1 - mag_lr)
        )
        self.policy = policy
        self.cum_policy += policy
        self.exp_avg_policy = 0.9 * self.exp_avg_policy + 0.1 * policy
        self.magnet = magnet

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""
        return self.policy.copy()

    def test_policy(self) -> np.ndarray:
        """Return policy for testing"""
        if self.test_magnet:
            return self.magnet.copy()
        elif self.test_average:
            return self.cum_policy / self.iteration
        else:
            return self.policy.copy()


class SmoothBR(Learner):
    def __init__(
        self,
        n_actions: int,
        temp: Callable[[int], float],
        noise: Callable[[np.ndarray], np.ndarray],
    ):
        """Implements of magnetic mirror descent

        Args:
            n_actions: Number of actions for the learner
            temp: Regularization temperature schedule

        """
        self.temp_fn = temp
        self.noise_fn = noise
        self.policy: np.ndarray = np.ones(n_actions) / n_actions
        self.iteration: int = 1
        self.temp = self.temp_fn(self.iteration)
        self.grad = np.array([0, 0])

    def update(self, qs: np.ndarray) -> None:
        """Update internal state of learner using `payoffs`"""
        self.iteration += 1
        self.temp = self.temp_fn(self.iteration)
        self.policy = softmax(self.noise_fn(qs) / self.temp)

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""
        return self.policy.copy()

    def test_policy(self, qs) -> np.ndarray:
        """Return policy for testing"""
        return softmax(qs / self.temp)
        # return self.policy.copy()


def make_dataset(game, temp: float, num_samples: int):
    p = np.random.uniform(0, 1, num_samples)
    ps_max = np.array([p, 1 - p]).T
    qs_min = game.compute_payoffs(ps_max, None)
    ps_min = softmax(qs_min / temp)
    qs_max = game.compute_payoffs(None, ps_min.T).T

    import matplotlib.pyplot as plt
    plt.scatter(ps_max[:, 0], qs_max[:, 0])
    plt.scatter(ps_max[:, 1], qs_max[:, 1])
    plt.show()
    print("a")

    return ps_max, ps_min, qs_max


class NearestNeighbour(Learner):
    def __init__(self, game, temp: float, num_samples: int, noise):
        self.temp = temp
        self.ps_max, self.ps_min, self.qs_max = make_dataset(game, temp, num_samples)
        self.qs_max = noise(self.qs_max)

        import matplotlib.pyplot as plt
        plt.scatter(self.ps_max[:, 0], self.qs_max[:, 0])
        plt.scatter(self.ps_max[:, 1], self.qs_max[:, 1])
        plt.show()

        self.grad = np.array([0, 0])
        self.policy = np.array([0.5, 0.5])

    def lookup(self, xs: np.ndarray) -> None:
        """ Find the closest ys to xs """
        idx = np.argmin(np.linalg.norm(self.ps_max - xs, axis=1))
        self.policy = self.ps_min[idx].ravel()
        return self.qs_max[idx]

    def train_policy(self) -> np.ndarray:
        """Return policy for learning"""
        return self.policy

    def test_policy(self) -> np.ndarray:
        """Return policy for testing"""
        return self.policy
