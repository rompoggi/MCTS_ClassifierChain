"""
Implementation of brute force algorithm to find the best label combination.
This explores the space of labels in a brute force manner to find the best label combination given the learnt probabilistic model.
The algorithm may not converge but it is guaranteed to find the best label combination given enough time, in the sense of the maximum likelihood estimation.
It may not however always find the exact label, as the classifier's error will propagate through the chain.
"""

from .mcts_config import MCTSConfig

import numpy as np
from typing import Tuple, Any
import threading
from datetime import datetime, timedelta
import multiprocessing as mp
from tqdm import tqdm


def BF(x, model, config: MCTSConfig) -> Any:  # pragma: no cover
    """
    Brute force algorithm to find the best label combination.

    Args:
        x (np.ndarray): The input data
        chain (ClassifierChain): The chain of classifiers to use
        n_classes (int): The number of classes

    Returns:
        Any: The best label combination
    """
    local = threading.local()

    class ExecutionTimeout(Exception):
        pass

    def start(d_time: float = 1.) -> None:
        max_duration = timedelta(seconds=d_time)
        local.start_time = datetime.now()
        local.max_duration = max_duration

    def check() -> None:
        if datetime.now() - local.start_time > local.max_duration:
            raise ExecutionTimeout()

    try:
        start(config.constraint.d_time)
        old_dict: dict[Tuple[int, ...], float] = {}
        new_dict: dict[Tuple[int, ...], float] = {}
        for i in range(config.n_classes):
            if i == 0:
                xy = x.reshape(1, -1)
                ps = model.estimators_[i].predict_proba(xy)[0]
                for j in range(2):
                    new_dict[(j,)] = ps[j]

                check()
            for k, v in old_dict.items():
                K = list(k)

                xy = np.concatenate([x, K]).reshape(1, -1)

                ps = model.estimators_[i].predict_proba(xy)[0]
                for j in range(2):
                    new_dict[tuple(K + [j])] = v * ps[j]

                check()

            old_dict = new_dict
            new_dict = {}

        return np.array(max(old_dict, key=lambda x: old_dict[x]))

    except ExecutionTimeout:
        return -1 * np.ones(config.n_classes)


def BF_wrapper(args) -> Any:
    return BF(*args)


def brute_force(x, model, config: MCTSConfig) -> Any:  # pragma: no cover
    """
    Brute force algorithm to find the best label combination.

    Args:
        x (np.ndarray): The input data
        chain (ClassifierChain): The chain of classifiers to use
        n_classes (int): The number of classes

    Returns:
        Any: The best label combination
    """
    if (config.constraint.time is False):
        raise ValueError("The constraint must be a time constraint")

    X = np.atleast_2d(x)

    if config.parallel:
        with mp.Pool(mp.cpu_count()) as pool:
            if config.verbose:
                out = list(tqdm(pool.imap(BF_wrapper, [(x, model, config) for x in X]), total=len(X)))
            else:
                out = pool.map(BF_wrapper, [(x, model, config) for x in X])

    else:
        if config.verbose:
            out = [BF(x, model, config) for x in tqdm(X, total=len(X))]
        else:
            out = [BF(x, model, config) for x in X]

    return np.atleast_2d(out)


__all__: list[str] = ["brute_force"]
