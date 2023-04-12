"""Interface for two-player zero-sum normal-form games.
"""
from typing import Optional

import numpy as np
from open_spiel.python.algorithms.mmd_dilated import neg_entropy


def dgf_eval(a, b):
    dgf_value = [0., 0.]
    probs = [a, b]
    for player in range(2):
        for p in probs[player]:
            dgf_value[player] += neg_entropy(p)
    return dgf_value


def entropy_sum(x):
    return -np.sum(x * np.log(x))

def smooth_br(qs, alpha):
    return np.exp(qs / alpha) / np.sum(np.exp(qs / alpha))


def softmax(x):
    # return np.exp(x) / np.sum(np.exp(x))
    unnormalized = np.exp(x - np.max(x))  # Why np.max(x)?
    return unnormalized / np.sum(unnormalized)


class Game:
    def __init__(self, payoff_table: np.ndarray):
        """Two-player zero-sum normal-form game

        Args:
            payoff_table: A 2-d array containing payoffs for joint actions
                Player 1's payoffs are equal to the entries, player 2's payoffs
                are equal to the negations of the entries

        Attributes:
            payoff_table: Array containing the payoffs for joint actions
            n_actions: The number of actions for each player
        """
        self.payoff_table: np.ndarray = payoff_table
        self.n_actions: tuple = self.payoff_table.shape

    def compute_payoffs(
            self, a: Optional[np.ndarray], b: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute action values for one player, given the other player's policy

        Args:
            a: Policy for player one
            b: Policy for player two
        Exactly one of `a` and `b` should be passed as input
        """
        if a is not None and b is None:
            #return -np.dot(a, self.payoff_table)
            return a @ self.payoff_table
        if a is None and b is not None:
            #return np.dot(self.payoff_table, b)
            return -self.payoff_table @ b
        raise ValueError

    def exploitability(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        """Compute exploitability

        Args:
            a: Policy for player one
            b: Policy for player two
        At least one of `a` and `b` should be passed as input
        """
        if a is None:
            return self.compute_payoffs(None, b).max()
        if b is None:
            return self.compute_payoffs(a, None).max()
        return (self.exploitability(a, None) + self.exploitability(None, b)) / 2

    def get_gap(self, x: np.ndarray, y: np.ndarray, alpha: float = 0.0) -> float:
        # if x and y are just 1d arrays, convert to column vectors
        if x.ndim == 1: x = x.reshape(-1, 1)
        if y.ndim == 1: y = y.reshape(-1, 1)
        ps = [x, y]

        # check probs are column vectors
        assert ps[0].shape == (self.n_actions[0], 1)
        assert ps[1].shape == (self.n_actions[1], 1)
        A = self.payoff_table
        grads = [A @ ps[1] / alpha, - ps[0].T @ A / alpha]
        brs = [softmax(-grads[0]).ravel(), softmax(-grads[1]).ravel()]

        dgf_values = [neg_entropy(ps[0]), neg_entropy(ps[1])]
        br_dgf_values = [neg_entropy(brs[0]), neg_entropy(brs[1])]
        # gap of policies (x,y)
        # d(x) + max_y' x.T A y'-d(y') + d(y) - min_x' d(x') + x'.T Ay

        gap = 0
        gap += ps[0].T @ A @ brs[1]
        gap += alpha * (dgf_values[1] - br_dgf_values[1])
        gap += alpha * (dgf_values[0] - br_dgf_values[0])
        gap += -brs[0].T @ A @ ps[1]
        return gap
