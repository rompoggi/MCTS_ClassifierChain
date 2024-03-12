"""
This module contains the implementation of the Monte Carlo Tree Search (MCTS) algorithm.
The MCTS algorithm is used to find the best child of a given node in a tree.
The tree is built by expanding the nodes and propagating the rewards back up the tree.

There are different policies to select the next node to visit, such as epsilon greedy here.
"""

import numpy as np
from typing import Any, Dict, Tuple, List
import multiprocessing as mp

from tqdm import tqdm

from .constraints import Constraint
from .mcts_node import MCTSNode, visualize_tree
from .mcts_config import MCTSConfig
from .policy import Policy, Uniform, Greedy


def select(node: MCTSNode, select_policy: Policy) -> MCTSNode:
    while (node.is_expanded() and not node.is_terminal()):
        idx: int = select_policy(node)
        node = node[idx]
    return node


def simulate(node: MCTSNode, simulate_policy: Policy = Uniform()) -> MCTSNode:
    while (not node.is_terminal()):
        if (not node.is_expanded()):
            node.expand()
        idx: int = simulate_policy(node)
        node = node[idx]
    return node


def get_reward(node: MCTSNode, x: Any, model: Any, cache: Dict[Tuple[int, ...], float] = {}, ys: List[int] = []) -> float:
    assert all(hasattr(est, 'predict_proba') for est in model.estimators_), "Model must have a predict_proba method"

    labels: List[int] = ys + node.get_parent_labels()
    if (tuple(labels)) in cache:
        return cache[tuple(labels)]

    assert (node.is_terminal()), f"Can only get rewards for a terminal node. Node rank={node.rank}."
    xy: np.ndarray[Any, np.dtype[Any]] = x.reshape(1, -1)
    p: float = 1.0

    for j in range(len(labels)):
        if j > 0:
            xy = np.column_stack([xy, labels[j-1]])

        p *= model.estimators_[j].predict_proba(xy)[0][labels[j]]

    cache[tuple(labels)] = p

    return p


def back_prog(node: MCTSNode, reward: float) -> None:
    if (node.parent is None):
        node.visit_count += 1
        return
    node.score = ((node.visit_count) * node.score + reward) / (node.visit_count + 1)
    node.visit_count += 1
    back_prog(node.parent, reward)


def best_child(node: MCTSNode, best_child_policy: Policy = Greedy()) -> int:
    return best_child_policy(node)


def best_child_all(root: MCTSNode, best_child_policy: Policy = Greedy()) -> list[int]:
    node: MCTSNode = root
    while (not node.is_terminal()):
        if (not node.is_expanded()):
            node.expand()
        node = node[best_child_policy(node)]
    return node.get_parent_labels()


def MCTS_one_step_atime(x, model, config) -> list[int]:  # pragma: no cover
    n_classes: int = config.n_classes
    ComputationalConstraint: Constraint = config.constraint

    select_policy: Policy = config.selection_policy
    simulate_policy: Policy = config.exploration_policy
    best_child_policy: Policy = config.best_child_policy

    ys: List[int] = []
    for k in range(n_classes):
        root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes-k, score=1.)
        cache: Dict[Tuple[int, ...], float] = {}

        ComputationalConstraint.reset()
        while (ComputationalConstraint):
            node: MCTSNode = select(root, select_policy=select_policy)
            leaf: MCTSNode = simulate(node, simulate_policy=simulate_policy)
            reward: float = get_reward(leaf, x, model, cache, ys=ys)
            back_prog(node, reward)

        bc: int = best_child(root, best_child_policy=best_child_policy)
        ys.append(bc)

        if config.visualize_tree_graph:
            visualize_tree(root, best_child=[bc], name=f"binary_tree_{k}", save=config.save_tree_graph)

    return ys


def MCTS_all_step_atime(x, model, config) -> list[int]:  # pragma: no cover
    n_classes: int = config.n_classes
    ComputationalConstraint: Constraint = config.constraint

    select_policy: Policy = config.selection_policy
    simulate_policy: Policy = config.exploration_policy
    best_child_policy: Policy = config.best_child_policy

    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes, score=1.)
    cache: Dict[Tuple[int, ...], float] = {}

    ComputationalConstraint.reset()
    while (ComputationalConstraint):
        node: MCTSNode = select(root, select_policy=select_policy)
        leaf = simulate(node, simulate_policy=simulate_policy)
        reward: float = get_reward(leaf, x, model, cache)
        back_prog(node, reward)

    bc: List[int] = best_child_all(root, best_child_policy=best_child_policy)

    if config.visualize_tree_graph:
        visualize_tree(root, best_child=bc, name="binary_tree", save=config.save_tree_graph)

    return bc


def MCTS_all_step_atime_wrapper(args) -> list[int]:
    return MCTS_all_step_atime(*args)


def MCTS_one_step_atime_wrapper(args) -> list[int]:
    return MCTS_one_step_atime(*args)


def MCTS(x, model, config: MCTSConfig) -> Any:  # pragma: no cover
    X = np.atleast_2d(x)

    if config.parallel:
        if config.step_once:
            MCTS_wrapper = MCTS_one_step_atime_wrapper
        else:
            MCTS_wrapper = MCTS_all_step_atime_wrapper

        with mp.Pool(mp.cpu_count()) as pool:
            if config.verbose:
                out = list(tqdm(pool.imap(MCTS_wrapper, [(x, model, config) for x in X]), total=len(X)))
            else:
                out = pool.map(MCTS_wrapper, [(x, model, config) for x in X])

    else:
        if config.step_once:
            func = MCTS_one_step_atime
        else:
            func = MCTS_all_step_atime

        if config.verbose:
            out = [func(x, model, config) for x in tqdm(X, total=len(X))]
        else:
            out = [func(x, model, config) for x in X]

    return np.atleast_2d(out)


__all__: list[str] = ["MCTS"]
