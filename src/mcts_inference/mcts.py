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
from typing import Any, Dict, Tuple

from .constraints import Constraint
from .mcts_node import MCTSNode, visualize_tree


def randmax(A: Any) -> int:
    """
    Function to return the index of the element with highest score in A.

    Args:
        A (Any): The list of MCTSNode

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node[1].score = 0.5
        >>> randmax(node.children)
        1
    """
    maxValue: Any = max(A)
    index: list[int] = [i for i in range(len(A)) if A[i] == maxValue]
    return int(np.random.choice(index))


def eps_greedy(node: MCTSNode, eps: float = 0.1) -> int:  # pragma: no cover
    """
    Epislon greedy policy to select the next node to visit.
    If eps=0, it is a greedy policy.

    Args:
        node (MCTSNode): The node from which to select the next node
        eps (float): The epsilon value for the epsilon greedy policy

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node[1].score = 0.5
        >>> eps_greedy(node, eps=0)
        1
        >>> eps_greedy(node, eps=1)
        0
        >>> eps_greedy(node, eps=1)
        1
    """
    assert (eps >= 0 and eps <= 1), f"{eps = } should be in the [0,1] range."
    if np.random.rand() < eps:  # explore
        return np.random.choice(node.n_children)
    return randmax(node.get_children_scores())


def ucb(node: MCTSNode, alpha: float = 0.5) -> int:  # pragma: no cover
    """
    Upper Confidence Bound (UCB) policy to select the next node to visit.

    Args:
        node (MCTSNode): The node from which to select the next node

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node[1].score = 0.5
        >>> ucb(node)
        1
    """
    assert (node.visit_count > 0), "Node has not yet been visited. A problem appened."

    if min([child.visit_count for child in node.children]) == 0:
        return randmax([-child.visit_count for child in node.children])

    ucb: np.ndarray[Any, np.dtype[Any]] = np.array([child.score + np.sqrt(alpha * np.log(node.visit_count) / child.visit_count) for child in node.children])
    return randmax(ucb)


def select(node: MCTSNode, eps: float = 0.2) -> MCTSNode:
    """
    Select the next node to visit using the MCTS algorithm.
    Selection is made using an epsilon greedy policy.

    Args:
        node (MCTSNode): The node from which to start the selection
        eps (float): The epsilon value for the epsilon greedy policy
    """
    while (node.is_expanded() and not node.is_terminal()):
        node.visit_count += 1
        ind: int = eps_greedy(node, eps)
        node = node[ind]
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


def simulate(node: MCTSNode, model: Any, x: Any, cache: Dict[Tuple[int, ...], float]) -> float:
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
        node = np.random.choice(np.array(node.children))  # uniform choice
        node.visit_count += 1
    return get_reward(node, model, x, cache)


def get_reward(node: MCTSNode, model: Any, x: Any, cache: Dict[Tuple[int, ...], float] = {}) -> float:  # pragma: no cover
    """
    Get the reward for the given node.
    The reward is obtained from the model that does the inference.

    Args:
        node (MCTSNode): The node for which to get the reward
        model (Any): The model to use for the reward evaluation, which has the predict_proba method
        x (Any): The input data
        cache (Dict[list[int], float]): The cache to store the reward evaluation
    """
    assert hasattr(model, 'predict_proba'), "Model must have a predict_proba method"

    labels: list[int] = node.get_parent_labels()
    if (tuple(labels)) in cache:
        return cache[tuple(labels)]

    assert (node.is_terminal()), f"Can only get rewards for a terminal node. Node rank={node.rank}."
    labels = node.get_parent_labels()
    xy: np.ndarray[Any, np.dtype[Any]] = x.reshape(1, -1)
    p: float = 1.0

    for j in range(len(labels)):
        if j > 0:
            # stack the previous y as an additional feature
            xy = np.column_stack([xy, labels[j-1]])

        p *= model.estimators_[j].predict_proba(xy)[0][labels[j]]  # (N.B. [0], because it is the first and only row)

    cache[tuple(labels)] = p

    return p


def best_child(root: MCTSNode) -> list[int]:  # best score
    """
    Returns the labels of the best child of the root node following a greedy policy.

    Args:
        root (MCTSNode): The root node from which to select the best child
    """
    return select(root, eps=0).get_parent_labels()


def MCTS(model, x, verbose: bool = False, secs: float = 1, visualize: bool = False, save: bool = False) -> list[int]:  # pragma: no cover
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
    n_classes: int = len(model.estimators_)
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes, score=1)

    ComputationalConstraint: Constraint = Constraint(time=True, d_time=secs, max_iter=False, n_iter=0, verbose=verbose)

    cache: Dict[Tuple[int, ...], float] = {}  # Create a cache to store the reward evaluation to gain inference speed
    while (ComputationalConstraint):
        node: MCTSNode = select(root)
        reward: float = simulate(node, model, x, cache)
        back_prog(node, reward)

    if visualize:
        bc: list[int] = best_child(root)
        visualize_tree(root, best_child=bc, name="binary_tree", save=save)
        return bc

    return best_child(root)


__all__: list[str] = ["MCTS", "randmax", "select", "back_prog",
                      "simulate", "get_reward", "best_child"]

if __name__ == "__main__":  # pragma: no cover
    from sklearn.datasets import make_multilabel_classification
    from sklearn.model_selection import train_test_split
    n_samples = 10000
    n_features = 6
    n_classes = 3
    n_labels = 2
    random_state = 0

    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=random_state)

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    from sklearn.multioutput import ClassifierChain
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear"
    base = LogisticRegression(solver=solver)
    chain: ClassifierChain = ClassifierChain(base)

    chain = chain.fit(X_train, Y_train)

    from tqdm import trange
    from sklearn.metrics import hamming_loss, zero_one_loss

    secs_lis: list[float] = [0.01, 0.1, 0.5, 1., 2.]

    M: int = min(100, len(Y_test))

    hl_mt: list[float] = []
    hl_ct: list[float] = []
    hl_mc: list[float] = []

    zo_mt: list[float] = []
    zo_ct: list[float] = []
    zo_mc: list[float] = []

    y_chain = chain.predict(X_test[:M])
    for secs in secs_lis:
        # continue
        _y_mcts = []

        for i in trange(M, desc=f"MCTS Inference Constraint={secs}s", unit="it", colour="green"):
            _y_mcts.append(MCTS(chain, X_test[i], secs=secs))

        y_mcts = np.array(_y_mcts)

        hl_mt.append(hamming_loss(y_mcts, Y_test[:M]))
        hl_ct.append(hamming_loss(y_chain, Y_test[:M]))
        hl_mc.append(hamming_loss(y_chain, y_mcts))

        zo_mt.append(zero_one_loss(y_mcts, Y_test[:M]))
        zo_ct.append(zero_one_loss(y_chain, Y_test[:M]))
        zo_mc.append(zero_one_loss(y_chain, y_mcts))

    import matplotlib.pyplot as plt

    plt.plot(secs_lis, hl_mt, label="MCTS vs True")
    plt.plot(secs_lis, hl_ct, label="Chains vs True")
    plt.plot(secs_lis, hl_mc, label="MCTS vs Chains")

    plt.title("Hamming Loss Comparison for different times")
    plt.xlabel("Seconds")
    plt.ylim(0, 1)
    plt.ylabel("Hamming Loss")
    plt.legend()
    plt.show()

    plt.plot(secs_lis, zo_mt, label="MCTS vs True")
    plt.plot(secs_lis, zo_ct, label="Chains vs True")
    plt.plot(secs_lis, zo_mc, label="MCTS vs Chains")

    plt.title("Zero One Loss Comparison for time different times")
    plt.xlabel("Seconds")
    plt.ylim(0, 1)
    plt.ylabel("Zero One Loss")
    plt.legend()
    plt.show()
