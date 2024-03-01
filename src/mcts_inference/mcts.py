"""
This module contains the implementation of the Monte Carlo Tree Search (MCTS) algorithm.
The MCTS algorithm is used to find the best child of a given node in a tree.
The tree is built by expanding the nodes and propagating the rewards back up the tree.

There are different policies to select the next node to visit, such as epsilon greedy here.

TODOS:
    - Add tests for main algorithm
    - Add a MCTS Config class to store the parameters of the MCTS algorithm
"""

import numpy as np
from typing import Any, Dict, Tuple, List
import multiprocessing as mp

from tqdm import tqdm

from .constraints import Constraint
from .mcts_node import MCTSNode, visualize_tree
from .mcts_config import MCTSConfig
from .utils import randmax
from .policy import Policy, Uniform, Greedy  # , EpsGreedy


def select(node: MCTSNode, policy: Policy) -> MCTSNode:
    """
    Select the next node to visit using the MCTS algorithm.
    Selection is made using an epsilon greedy policy.

    Args:
        node (MCTSNode): The node from which to start the selection
        eps (float): The epsilon value for the epsilon greedy policy
    """
    while (node.is_expanded() and not node.is_terminal()):
        node.visit_count += 1
        idx: int = policy(node)
        node = node[idx]
    return node


def back_prog(node: MCTSNode, reward: float) -> None:
    """
    Propagate the reward back up the tree.

    Args:
        node (MCTSNode): The node from which to propagate the reward
        reward (float): The reward to propagate

    Returns:
        None
    """
    if node.parent is None:  # root node, no need to update
        return
    assert (node.visit_count > 0), "Node has not yet been visited. A problem appened."
    node.score += reward
    back_prog(node.parent, reward)


def simulate(node: MCTSNode, policy: Policy = Uniform()) -> MCTSNode:
    """
    Simulate the rest of the episode from the given node.
    Returns the reward for the episode.

    Args:
        node (MCTSNode): The node from which to simulate the episode
        model (Any): The model to use for the simulation
        x (Any): The input data
        cache (Dict[tuple[int], float]): The cache to store the reward evaluation

    Returns:
        float: The reward for the episode
    """
    node.visit_count += 1
    while (not node.is_terminal()):
        if not node.is_expanded():
            node.expand()
        idx: int = policy(node)
        node = node[idx]
        node.visit_count += 1
    return node


def get_reward(node: MCTSNode, model: Any, x: Any, cache: Dict[Tuple[int, ...], float] = {}, ys: List[int] = []) -> float:
    """
    Get the reward for the given node.
    The reward is obtained from the model that does the inference.

    Args:
        node (MCTSNode): The node for which to get the reward
        model (Any): The model to use for the reward evaluation, which has the predict_proba method
        x (Any): The input data
        cache (Dict[list[int], float]): The cache to store the reward evaluation
    """
    assert all(hasattr(est, 'predict_proba') for est in model.estimators_), "Model must have a predict_proba method"

    labels: List[int] = ys + node.get_parent_labels()
    print(labels)
    if (tuple(labels)) in cache:
        return cache[tuple(labels)]

    assert (node.is_terminal()), f"Can only get rewards for a terminal node. Node rank={node.rank}."
    xy: np.ndarray[Any, np.dtype[Any]] = x.reshape(1, -1)
    p: float = 1.0

    for j in range(len(labels)):
        if j > 0:
            # stack the previous y as an additional feature
            xy = np.column_stack([xy, labels[j-1]])

        p *= model.estimators_[j].predict_proba(xy)[0][labels[j]]  # (N.B. [0], because it is the first and only row)

    cache[tuple(labels)] = p

    return p


def best_child(node: MCTSNode, policy: Policy = Greedy()) -> int:
    """
    Returns the labels of the best child of the root node following a greedy policy.

    Args:
        node (MCTSNode): The root node from which to select the best child
        policy (Policy): The policy to use to select the best child
    """
    res: int | None = node[policy(node)].label
    return res if (res is not None) else -1


def best_child_all(root: MCTSNode, policy: Policy = Greedy()) -> list[int]:
    """
    Returns the labels of the best child of the root node following a greedy policy.

    Args:
        root (MCTSNode): The root node from which to select the best child
    """
    node: MCTSNode = root
    while (not node.is_terminal()):
        node = node[policy(node)]
    return node.get_parent_labels()


def _one_step_MCTS(model, x, config) -> list[int]:  # pragma: no cover
    """
    Perform the Monte Carlo Tree Search (MCTS) algorithm step by step.

    Args:
        model: The machine learning model used for inference.
        x: The input data for inference.
        config: The configuration object containing various MCTS parameters.

    Returns:
        List[int]: The predicted labels for the input data.

    """
    n_classes: int = config.n_classes
    ComputationalConstraint: Constraint = config.constraint

    select_policy: Policy = config.selection_policy
    simulate_policy: Policy = config.exploration_policy
    best_child_policy: Policy = config.best_child_policy

    ys: List[int] = []
    for k in range(n_classes):
        root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes-k, score=1.)
        cache: Dict[Tuple[int, ...], float] = {}  # Create a cache to store the reward evaluation to gain inference speed

        ComputationalConstraint.reset()
        while (ComputationalConstraint):
            node: MCTSNode = select(root, policy=select_policy)
            node = simulate(node, policy=simulate_policy)
            reward: float = get_reward(node, model, x, cache, ys=ys)
            back_prog(node, reward)

        bc: int = best_child(root, policy=best_child_policy)
        ys.append(bc)

        if config.visualize_tree_graph:
            visualize_tree(root, best_child=[bc], name=f"binary_tree_{k}", save=config.save_tree_graph)

    return ys


def _all_step_MCTS(model, x, config) -> list[int]:  # pragma: no cover
    n_classes: int = config.n_classes
    ComputationalConstraint: Constraint = config.constraint

    select_policy: Policy = config.selection_policy
    simulate_policy: Policy = config.exploration_policy
    best_child_policy: Policy = config.best_child_policy

    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes, score=1.)
    cache: Dict[Tuple[int, ...], float] = {}  # Create a cache to store the reward evaluation to gain inference speed

    ComputationalConstraint.reset()
    while (ComputationalConstraint):
        node: MCTSNode = select(root, policy=select_policy)
        node = simulate(node, policy=simulate_policy)
        reward: float = get_reward(node, model, x, cache)
        back_prog(node, reward)

    bc: List[int] = best_child_all(root, policy=best_child_policy)

    if config.visualize_tree_graph:
        visualize_tree(root, best_child=bc, name="binary_tree", save=config.save_tree_graph)

    return bc


def _all_step_MCTS_wrapper(args) -> list[int]:
    return _all_step_MCTS(*args)


def _one_step_MCTS_wrapper(args) -> list[int]:
    return _one_step_MCTS(*args)


def MCTS(model, X, config: MCTSConfig) -> Any:  # pragma: no cover
    """
    Monte Carlo Tree Search alogrithm.

    Args:
        model (Any): The model to use for the MCTS algorithm
        x (Any): The input data
        verbose (bool): If True, the constraints will print a message when they are reached
        secs (float): The time constraint in seconds
        visualize (bool): If True, the search tree will be visualized

    Returns:
        list[int]: The labels of the best child of the root node following a greedy policy
    """
    X = np.atleast_2d(X)

    if config.parallel:
        if config.step_once:
            MCTS_wrapper = _one_step_MCTS_wrapper
        else:
            MCTS_wrapper = _all_step_MCTS_wrapper

        with mp.Pool(mp.cpu_count()) as pool:
            if config.verbose:
                out = list(tqdm(pool.imap(MCTS_wrapper, [(model, x, config) for x in X]), total=len(X)))
            else:
                out = pool.map(MCTS_wrapper, [(model, x, config) for x in X])

    else:
        if config.step_once:
            func = _one_step_MCTS
        else:
            func = _all_step_MCTS

        if config.verbose:
            out = [func(model, x, config) for x in tqdm(X, total=len(X))]
        else:
            out = [func(model, x, config) for x in X]

    return np.atleast_2d(out)


__all__: list[str] = ["MCTS", "randmax", "select", "back_prog",
                      "simulate", "get_reward", "best_child", "best_child_all",
                      "_all_step_MCTS", "_all_step_MCTS_wrapper",
                      "_one_step_MCTS", "_one_step_MCTS_wrapper"
                      ]
