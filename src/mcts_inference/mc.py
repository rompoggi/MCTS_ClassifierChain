"""
This module contains the implementation of the Monte Carlo Search for the Classifier Chain algorithm.
It is refered to Probabilistic Classifier Chains in the literature.
See Efficient Monte Carlo Methods for Multi-Dimensional Learning with Classifier Chains by
Jesse Read, Luca Martino, David Luengo for more information. https://arxiv.org/abs/1211.2190
"""

from .constraints import Constraint

from typing import List, Tuple, Optional, Any, TypeVar
import multiprocessing as mp
import numpy as np
from tqdm import tqdm


T = TypeVar('T', bound='MCNode')  # Define the type now to not have an issue with recursive typing


class MCNode:
    def __init__(self,
                 label: Optional[int] = 0,
                 rank: int = 2,
                 n_children: int = 2,
                 score: float = 0.,
                 parent_labels: List[int] = []) -> None:

        self.score: float = score
        self.label: Optional[int] = label

        self.children: List[MCNode] = []
        self.n_children: int = n_children
        self.rank: int = rank

        self.parent_labels: List[int] = parent_labels

    def __getitem__(self, key: int) -> "MCNode":
        assert (key >= 0 and key < self.n_children), f"{key} is not a valid key."
        assert (self.is_expanded()), f"Node not yet expanded. Cannot get the child node at key:{key}."
        return self.children[key]

    def is_terminal(self) -> bool:
        return (self.rank == 0)

    def is_expanded(self) -> bool:
        return (len(self.children) != 0)

    def expand(self, x, model) -> None:
        assert all(hasattr(est, 'predict_proba') for est in model.estimators_), "Model must have a predict_proba method"
        assert (not self.is_terminal()), "Cannot expand a terminal node"
        assert (not self.is_expanded()), "Node already expanded. Cannot expand again."
        self.children = [MCNode(label=i, rank=self.rank-1, n_children=self.n_children, parent_labels=self.parent_labels+[i]) for i in range(self.n_children)]
        labels: List[int] = self.get_parent_labels()

        xy: np.ndarray[Any, np.dtype[Any]] = np.concatenate([x, labels])
        xy = xy.reshape(1, -1)
        ps = model.estimators_[-self.rank].predict_proba(xy)[0]
        for p, child in zip(ps, self.children):
            child.score = p

    def get_parent_labels(self) -> List[int]:
        return self.parent_labels

    def is_fully_expanded(self) -> bool:
        if self.is_terminal():
            return True
        for child in self.children:
            if not child.is_fully_expanded():
                return False
        return self.is_expanded()

    def get_children_scores(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        assert (self.is_expanded()), "Node not yet expanded. Cannot get the children scores."
        return np.array([child.score for child in self.children], dtype=np.float64)

#############################


def mc_simulate(node: MCNode, x, model) -> Tuple[MCNode, float]:
    """
    Simulate the rest of the episode from the given node.
    Returns the reward for the episode.

    Args:
        node (MCNode): The node from which to simulate the episode
        model (Any): The model to use for the simulation
        x (Any): The input data
        cache (Dict[tuple[int], float]): The cache to store the reward evaluation

    Returns:
        float: The reward for the episode
    """
    p: float = 1.
    while (not node.is_terminal()):
        if not node.is_expanded():
            node.expand(x=x, model=model)
        idx: int = np.random.choice(a=node.n_children, p=node.get_children_scores())
        node = node[idx]
        p *= node.score
    return node, p


def _MC(x, model, config, get_root: List[MCNode] = []) -> list[int]:  # pragma: no cover
    n_classes: int = config.n_classes
    ComputationalConstraint: Constraint = config.constraint

    root: MCNode = MCNode(label=None, n_children=2, rank=n_classes, score=1.)

    best_score: float = - np.inf
    best_child: List[int] = [0] * n_classes

    ComputationalConstraint.reset()
    while (ComputationalConstraint):
        node, score = mc_simulate(root, x, model)
        if score > best_score:
            best_score = score
            best_child = node.get_parent_labels()

    get_root.append(root)
    return best_child


def MC_wrapper(args) -> list[int]:
    return _MC(*args)


def MC(X, model, config, get_root: List[MCNode] = []) -> Any:  # pragma: no cover
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
        with mp.Pool(mp.cpu_count()) as pool:
            if config.verbose:
                out = list(tqdm(pool.imap(MC_wrapper, [(x, model, config) for x in X]), total=len(X)))
            else:
                out = pool.map(MC_wrapper, [(x, model, config) for x in X])

    else:
        if config.verbose:
            out = [_MC(x, model, config, get_root=get_root) for x in tqdm(X, total=len(X))]
        else:
            out = [_MC(x, model, config, get_root=get_root) for x in X]

    return np.atleast_2d(out)


__all__: list[str] = ["MCNode", "MC"]
