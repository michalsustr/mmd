"""Interface to run experiments for two-player zero-sum normal-form games.
"""
from collections import defaultdict
from typing import Union

import pandas as pd
from tqdm import tqdm

from .game import Game, softmax
from .learners import Learner, NearestNeighbour, MMD
from ..utils import kl, schedule


def main(
    game: Game,
    maximizer: Learner,
    minimizer: Learner,
    iterations: int,
    file: str,
) -> None:
    """Run `maximizer` against `minimizer` in `game`

    Args:
        game: Game in which to train agents
        maximizer: Agent with maximizing objective
        minimizer: Agent with minimizing objective
        iterations: Number of iterations to run
        file: Filename to which to save data
    """
    data = defaultdict(list)
    for i, should_save in tqdm(schedule(iterations), desc="play iterations"):
        data["it"] += [i]
        max_ps = maximizer.test_policy()
        qs = game.compute_payoffs(max_ps, None)
        min_ps = minimizer.test_policy(qs)
        data["expl"] += [game.exploitability(max_ps, min_ps)]
        data["gap_a"] += [float(game.get_gap(max_ps, min_ps, alpha=maximizer.temp))]
        data["p1"] += [max_ps[0]]
        data["p2"] += [min_ps[0]]
        data["p2_train"] += [minimizer.train_policy()[0]]
        data["div1"] += [maximizer.grad.sum()]
        data["div2"] += [minimizer.grad.sum()]
        data["temp1"] += [maximizer.temp]
        data["temp2"] += [minimizer.temp]
        maximizer.update(game.compute_payoffs(None, minimizer.train_policy()))
        minimizer.update(game.compute_payoffs(maximizer.train_policy(), None))
        if should_save:
            df = pd.DataFrame(data)
            df.to_csv(file + ".csv")


def main_neigh(
    game: Game,
    maximizer: MMD,
    lookup: NearestNeighbour,
    iterations: int,
    file: str,
) -> None:
    """Run `maximizer` against `minimizer` in `game`

    Args:
        game: Game in which to train agents
        maximizer: Agent with maximizing objective
        minimizer: Agent with minimizing objective
        iterations: Number of iterations to run
        file: Filename to which to save data
    """
    data = defaultdict(list)
    for i, should_save in tqdm(schedule(iterations), desc="play iterations"):
        data["it"] += [i]
        max_ps = maximizer.test_policy()
        min_ps = lookup.train_policy()
        min_ps_test = softmax(game.compute_payoffs(maximizer.train_policy(), None) / lookup.temp)
        data["expl"] += [game.exploitability(max_ps, min_ps)]
        data["expl_magnet"] += [game.exploitability(maximizer.magnet, softmax(game.compute_payoffs(maximizer.magnet, None) / lookup.temp))]
        avg = maximizer.cum_policy / maximizer.iteration
        data["expl_avg"] += [game.exploitability(avg, softmax(game.compute_payoffs(avg, None) / lookup.temp))]
        data["expl_exp_avg"] += [game.exploitability(maximizer.exp_avg_policy, softmax(game.compute_payoffs(maximizer.exp_avg_policy, None) / lookup.temp))]
        data["gap_a"] += [float(game.get_gap(max_ps, min_ps, alpha=maximizer.temp))]
        data["p1"] += [max_ps[0]]
        data["p2"] += [min_ps_test[0]]
        data["p2_train"] += [min_ps[0]]
        data["div1"] += [maximizer.grad.sum()]
        data["div2"] += [lookup.grad.sum()]
        data["temp1"] += [maximizer.temp]
        data["temp2"] += [lookup.temp]
        maximizer.update(lookup.lookup(maximizer.train_policy()))
        if should_save:
            df = pd.DataFrame(data)
            df.to_csv(file + ".csv")
