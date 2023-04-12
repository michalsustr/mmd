import math
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from numba import njit
from tqdm import tqdm

from mmd.nfg.game import Game, softmax
from mmd.nfg.learners import MMD, SmoothBR, NearestNeighbour
from mmd.nfg.run import main, main_neigh


def perturbed_rps():
    # The matrix is
    #  0 -3 3
    #  3 0 -1
    #  -3 1 0
    # The value is 0.
    # An optimal strategy for Player I is:
    #  (0.14286,0.42857,0.42857)
    # An optimal strategy for Player II is:
    #  (0.14286,0.42857,0.42857)
    return Game(normalize(np.array([[0, -3, 3], [3, 0, -1], [-3, 1, 0]])))

def biased_mp():
    # The matrix is
    #  1 -1 0
    #  -2 1 0
    # The value is -0.2.
    # An optimal strategy for Player I is:
    #  (0.6,0.4)
    # An optimal strategy for Player II is:
    #  (0.4,0.6,0)
    return Game(np.array([
        [ 1,-1],
        [-12, 1],
    ]))


@njit
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def compute_alpha_reg(alpha, iters=None):
    """ Find the alpha-reg equilibrium. """
    players = [
        MMD(
            n_actions=n_action,
            initializer=lambda t: np.ones(n_action) / n_action,
            lr=lambda t: alpha,
            temp=lambda t: alpha,
            magnet_initializer=lambda t: np.ones(n_action) / n_action,
            magnet_lr=lambda t: 0,
            use_magnet=False,
            noise=lambda qs: qs,
        )
        for n_action in n_actions
    ]
    maximizer = players[0]
    minimizer = players[1]
    if iters is None:
        iters = int(1000 * 1/alpha)  # TODO: find proper number of iterations for epsilon error

    for _ in range(iters):
        maximizer.update(game.compute_payoffs(None, minimizer.train_policy()))
        minimizer.update(game.compute_payoffs(maximizer.train_policy(), None))
    return maximizer.train_policy(), minimizer.train_policy()


def precompute_alpha_reg_trajectory(alphas):
    for a in tqdm(alphas, desc="precompute_alpha_reg_trajectory"):
        x, y = compute_alpha_reg(a)
        yield x, y, float(game.get_gap(x, y, alpha=a))


def make_plot(file: str):
    df = pd.read_csv(file + ".csv")
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10),
                             # return axes as 2d
                             squeeze=False
                             )
    axes[0, 0].loglog(df["it"], df["expl"])
    axes[0, 0].loglog(df["it"], df["expl_magnet"])
    axes[0, 0].loglog(df["it"], df["expl_avg"])
    axes[0, 0].loglog(df["it"], df["expl_exp_avg"])
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("exploitability")
    # axes[0, 1].plot(df["p1"], df["p2"], c=df["it"])
    # Plot with color as function of iteration
    # axes[0, 1].scatter(df["p1"], df["p2"], c=df["it"])

    axes[0, 1].loglog(df["it"], df["gap_a"])
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("gap_a(t)")

    ax = axes[0, 1].twinx()
    ax.loglog(df["it"], df["temp1"], c="r")
    ax.set_ylabel("alpha", color="r")

    axes[1, 0].semilogx(df["it"], df["p1"])
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("p1(heads)")
    axes[1, 0].set_ylim((0, 1))
    # ax = axes[1, 0].twinx()
    # ax.plot(df["it"], df["div1"], c="r")
    # ax.set_ylabel("div_1")

    axes[1, 1].semilogx(df["it"], df["p2"])
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("p2_test(heads)")
    axes[1, 1].set_ylim((0, 1))

    # ax = axes[1, 1].twinx()
    # ax.plot(df["it"], df["p2_train"], c="r")
    # ax.set_ylabel("p2_train(heads)")

    # ax = axes[1, 1].twinx()
    # ax.plot(df["it"], df["div2"], c="r")
    # ax.set_ylabel("div_2")

    x = df["p1"]
    y = df["p2"]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis')
    lc.set_linewidth(2)
    # cols = np.log(df["temp1"])
    cols = np.log(df["it"])
    lc.set_array(cols)
    line = axes[2, 0].add_collection(lc)
    # fig.colorbar(line, ax=axes[2, 0], label="log(alpha)")
    fig.colorbar(line, ax=axes[2, 0], label="it")

    axes[2, 0].set_ylim((0, 1))
    axes[2, 0].set_xlim((0, 1))
    axes[2, 0].set_xlabel("p1(heads)")
    axes[2, 0].set_ylabel("p2_test(heads)")


    n = 10000
    ps = np.linspace(0, 1, n)
    xs = np.array([ps, 1 - ps])
    ys = np.array([ps, 1 - ps])
    brv1 = game.compute_payoffs(None, ys).max(axis=0).reshape((n, 1))
    brv2 = game.compute_payoffs(xs.T, None).max(axis=1).reshape((1, n))
    expl = (brv1 + brv2) / 2
    axes[2, 0].contour(ps, ps, expl, levels=30, cmap="gray", linewidths=0.5, linestyles="dashed")
    axes[2, 0].set_aspect("equal")
    print("save 1: ", file + ".pdf")
    plt.savefig(file + ".pdf")

    alphas = np.logspace(-1, 2, 100)[::-1]
    precomputed_table = list(precompute_alpha_reg_trajectory(alphas))
    x = [rec[0][0] for rec in precomputed_table]
    y = [rec[1][0] for rec in precomputed_table]
    gaps = [rec[2] for rec in precomputed_table]
    axes[2, 0].plot(x, y, label="precomputed", color="black", linewidth=2)

    # axes[2, 1].loglog(alphas, gaps)
    # axes[2, 1].set_xlabel("alpha")
    # axes[2, 1].set_ylabel("gap_alpha")

    plt.tight_layout()
    print("save 2: ", file + ".pdf")
    plt.savefig(file + ".pdf")


def uniform(n):
    return np.ones(n) / n


def self_play_anneal_magnet_and_temp():
    return [
        MMD(
            n_actions=n_action,
            initializer=uniform,

            temp=lambda t: 1 / math.sqrt(t),
            lr=lambda t: 1.99 * (1 / math.sqrt(t)) / (game.payoff_table.max() ** 2),

            magnet_initializer=uniform,
            magnet_lr=lambda t: 1 / t,
            use_magnet=True,
        ) for n_action in n_actions
    ]


noise_table_lookup = defaultdict(
    lambda: np.random.uniform(-.1 * u_max, .1 * u_max, size=[n_actions[0]])
    )


@njit
def noise_fn(qs: np.ndarray) -> np.ndarray:
    return qs + np.random.uniform(-noise * u_max, noise * u_max, size=qs.shape)


@njit
def anneal_lr(t: float):
    return 1.99 * (1 / math.sqrt(t)) / (u_max ** 2)


def play_against_br():
    n_action = n_actions[0]
    maxim = MMD(
            n_actions=n_action,
            initializer=uniform,

            temp=lambda t: temp,
            # temp=lambda t: 1 / math.sqrt(t),
            lr=anneal_lr,
            # lr=lambda t: temp,

            magnet_initializer=uniform,
            magnet_lr=lambda t: 1 / t,
            use_magnet=False,
            test_magnet=False,
            test_average=False,

            # noise=lambda qs: qs
            noise=noise_fn,
            # noise=lambda qs: qs + noise_table_lookup[tuple(np.round(qs, 4))],
        )
    br = SmoothBR(n_actions=n_actions[1],
                  # temp=lambda t: 1 / math.sqrt(t),
                  temp=lambda t: temp,
                  noise=lambda qs: qs
                  # noise=lambda qs: qs + noise_table_lookup[tuple(np.round(qs, 4))],
                  # noise=lambda qs: qs + np.random.uniform(-noise * u_max, noise * u_max, size=qs.shape)
                  )
    return [maxim, br]


def play_against_nearest_neighbour():
    n_action = n_actions[0]
    maxim = MMD(
            n_actions=n_action,
            initializer=uniform,

            temp=lambda t: temp,
            # temp=lambda t: 1 / math.sqrt(t),
            lr=anneal_lr,
            # lr=lambda t: temp,

            magnet_initializer=uniform,
            magnet_lr=lambda t: 1 / t,
            use_magnet=False,
            test_magnet=False,
            test_average=False,

            # noise=lambda qs: qs
            noise=noise_fn,
            # noise=lambda qs: qs + noise_table_lookup[tuple(np.round(qs, 4))],
        )

    biased_noise_fn = lambda qs: qs + np.random.normal(0.1, noise, size=qs.shape)
    br = NearestNeighbour(game, temp, 100000,
                          noise=biased_noise_fn,
                          )
    return [maxim, br]


if __name__ == "__main__":

    game = biased_mp()
    u_max = game.payoff_table.max()
    n_actions = game.payoff_table.shape

    noise = .1
    temp = .1

    # players = self_play_anneal_magnet_and_temp()
    # players = play_against_br()
    players = play_against_nearest_neighbour()
    iterations = 10000
    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/nfg"
    Path(directory).mkdir(exist_ok=True, parents=True)
    file = directory + f"/result"
    main_neigh(
        game=game,
        maximizer=players[0],
        lookup=players[1],
        iterations=iterations,
        file=file,
    )
    make_plot(file)

