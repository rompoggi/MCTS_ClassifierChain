"""
This module contains the implementation of the Monte Carlo Tree Search (MCTS) algorithm.
The MCTS algorithm is used to find the best child of a given node in a tree.
The tree is built by expanding the nodes and propagating the rewards back up the tree.

There are different policies to select the next node to visit, such as epsilon greedy here.

TODOS:
    - Add more examples
    - Add more tests

    - Implement more Policies
"""

import numpy as np
from typing import Optional, Any, Dict, List, TypeVar

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

from mcts_inference.constraints import Constraint
from mcts_inference.utils import NormOption


T = TypeVar('T', bound='MCTSNode')  # Define the type now to not have an issue with recursive typing


class MCTSNode:
    """
    MCTSNode class to represent a node in the MCTS algorithm.

    Args:
        label (Optional[int]): The label of the node
        rank (int): The rank of the node
        n_children (int): The number of children for the node
        proba (float): The probability of the node
        parent (Optional[MCTSNode]): The parent node
        parent_labels (List[int]): The labels of the parent nodes

    Attributes:
        proba (float): The probability of the node
        label (Optional[int]): The label of the node
        visit_count (int): The number of times the node has been visited

        parent (Optional[MCTSNode]): The parent node
        children (List[MCTSNode]): The children nodes
        n_children (int): The number of children for the node
        rank (int): The rank of the node

        parent_labels (List[int]): The labels of the parent nodes

    Methods:
        __get__(self, key: int) -> "MCTSNode": get the child node at the given key
        is_terminal(self) -> bool: Returns True if the node is a terminal node, False otherwise
        is_expanded(self) -> bool: Returns True if the node is expanded, False otherwise
        expand(self) -> None: Expand the node by creating its children
        get_parent_labels(self) -> list[int]: Get the labels of the parent nodes
        __str__(self) -> str: String representation of the node
        __repr__(self) -> str: String representation of the node
        print_all(self) -> None: Print all the attributes of the node in a readable format
        delete(self) -> None: Delete the node and all its children
        check_correctness(self) -> bool: Checks recursively that all the children's visit count sum to that of their parent node
        is_fully_expanded(self) -> bool: Checks recursively if the entire tree has been expanded
        normalize_proba(self, opt: NormOption = NormOption.SOFTMAX) -> None: Normalizes the rewards obtained at each node into a distribution

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
        >>> node
        (MCTSNode: L=0, R=2, P=0.5000, PL[])
        >>> node.is_terminal()
        False
        >>> node.is_expanded()
        False
        >>> node.expand()
        >>> node.is_expanded()
        True
        >>> node.children
        [(MCTSNode: L=0, R=1, P=0.0000, PL[0]), (MCTSNode: L=1, R=1, P=0.0000, PL[1])]
        >>> node.children[0].parent
        (MCTSNode: L=0, R=2, P=0.5000, PL[])
    """
    def __init__(self,
                 label: Optional[int] = 0,
                 rank: int = 2,
                 n_children: int = 2,
                 proba: float = 0.,
                 parent: Optional["MCTSNode"] = None,
                 parent_labels: List[int] = []) -> None:

        self.proba: float = proba
        self.label: Optional[int] = label
        self.visit_count: int = 0

        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.n_children: int = n_children
        self.rank: int = rank

        self.parent_labels: List[int] = parent_labels

    def __get__(self, key: int) -> "MCTSNode":
        """
        get the child node at the given key
        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node[0]
            (MCTSNode: L=0, R=1, P=0.0000, PL[0])
            >>> node[1]
            (MCTSNode: L=1, R=1, P=0.0000, PL[1])
        """
        assert (key >= 0 and key < self.n_children), f"{key} is not a valid key."
        assert (self.is_expanded()), f"Node not yet expanded. Cannot get the child node at key:{key}."
        return self.children[key]

    def is_terminal(self) -> bool:
        """
        Returns True if the node is a terminal node, False otherwise.
        A node is terminal if its rank is 0.
        """
        return (self.rank == 0)

    def is_expanded(self) -> bool:
        """
        Returns True if the node is expanded, False otherwise.
        A node is expanded if it has children.
        """
        return (len(self.children) != 0)

    def expand(self) -> None:
        """
        Expand the node by creating its children.
        Each child will have a rank one less than the parent node.
        """
        assert (not self.is_terminal()), "Cannot expand a terminal node"
        self.children = [MCTSNode(label=i, rank=self.rank-1, n_children=self.n_children,
                                  parent=self, parent_labels=self.parent_labels+[i]) for i in range(self.n_children)]

    def get_parent_labels(self) -> list[int]:
        """
        Get the labels of the parent nodes.
        """
        return self.parent_labels

    def __str__(self) -> str:
        """
        String representation of the node
        """
        out: str = f"(MCTSNode: L={self.label}, R={self.rank}, P={self.proba:.4f}, PL{self.parent_labels})"
        return out

    def __repr__(self) -> str:
        """
        String representation of the node
        """
        return str(self)

    def print_all(self) -> None:
        """
        Print all the attributes of the node in a readable format.
        """
        print("\n".join([f"{k}:{v}" for k, v in self.__dict__.items()]))

    def delete(self) -> None:
        """
        Delete the node and all its children.
        """
        # if not self.is_expanded() :  # Code never reached
        #     del self
        #     return
        for child in self.children:
            child.delete()
        del self

    def check_correctness(self) -> bool:
        """
        Checks recursively that all the children's visit count sum to that of their parent node.

        Args:
            node (MCTSNode): The node from which to start the check

        Returns:
            bool: True if the visit count sum is correct, False otherwise

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node.children[0].visit_count = 2  # simulate that child 0 has been visited twice
            >>> node.children[1].visit_count = 3  # simulate that child 1 has been visited three times
            >>> node.visit_count
            0
            >>> node.check_correctness()
            False
            >>> node.visit_count = 5
            >>> node.check_correctness()
            True
        """
        # if self.children is None:  # Code never reached
        #     return True
        ssum: int = 0
        for child in self.children:
            if not child.check_correctness():
                return False
            ssum += child.visit_count
        return (ssum == self.visit_count)

    def is_fully_expanded(self) -> bool:
        """
        Checks recursively if the entire tree has been expanded

        Args:
            node (MCTSNode): The node from which to start the check

        Returns:
            bool: True if the entire tree has been expanded, False otherwise
        """
        if self.is_terminal():
            return True
        for child in self.children:
            if not child.is_fully_expanded():
                return False
        return True

    def normalize_proba(self, opt: NormOption = NormOption.SOFTMAX) -> None:
        """
        Normalizes the rewards obtained at each node into a distribution

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node.children[0].proba = 0.
            >>> node.children[1].proba = 2.
            >>> node.normalize_proba(opt=NormOption.SOFTMAX)
            >>> node.children[0].proba
            0.0
            >>> node.children[1].proba
            1.0
        """
        if self.is_terminal():
            return
        probas: np.ndarray[Any, np.dtype[Any]] = np.array([child.proba for child in self.children])

        if opt == NormOption.SOFTMAX:
            probas = np.exp(probas)
        elif opt == NormOption.UNIFORM:
            pass
        elif opt == NormOption.NONE:
            pass  # Do nothing

        if opt != NormOption.NONE:
            probas /= np.sum(probas)

        for i, child in enumerate(self.children):
            child.proba = probas[i]


# @debug
def randmax(A: Any) -> int:
    """
    Function to return the index of the element with highest proba in A.

    Args:
        A (Any): The list of MCTSNode

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node.children[1].proba = 0.5
        >>> randmax(node.children)
        1
    """
    maxValue: list[MCTSNode] = max(A, key=lambda x: x.proba).proba
    index: list[int] = [i for i in range(len(A)) if A[i].proba == maxValue]
    return int(np.random.choice(index))


# @debug
def eps_greedy(node: MCTSNode, eps: float = 0.1) -> int:
    """
    Epislon greedy policy to select the next node to visit.
    If eps=0, it is a greedy policy.

    Args:
        node (MCTSNode): The node from which to select the next node
        eps (float): The epsilon value for the epsilon greedy policy

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, proba=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node.children[1].proba = 0.5
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
    return randmax(node.children)


# @debug
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
        node = node.children[ind]
    return node


# @debug
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
    node.proba = node.proba + (reward - node.proba) / node.visit_count  # average proba
    back_prog(node.parent, reward)


# @debug
def simulate(node: MCTSNode, model: Any, x: Any, cache) -> float:
    """
    Simulate the rest of the episode from the given node.
    Returns the reward for the episode.

    Args:
        node (MCTSNode): The node from which to simulate the episode
        model (Any): The model to use for the simulation
        x (Any): The input data
        cache (Dict[list[int], float]): The cache to store the reward evaluation

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


# @debug
def get_reward(node: MCTSNode, model: Any, x: Any, cache: Dict[list[int], float] = {}) -> float:
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
    if (labels) in cache:
        return cache[labels]

    assert (node.is_terminal()), f"Can only get rewards for a terminal node. Node rank={node.rank}."
    labels = node.get_parent_labels()
    xy: np.ndarray[Any, np.dtype[Any]] = x.reshape(1, -1)
    p: float = 1.0

    for j in range(len(labels)):
        if j > 0:
            # stack the previous y as an additional feature
            xy = np.column_stack([xy, labels[j-1]])

        p *= model.estimators_[j].predict_proba(xy)[0][labels[j]]  # (N.B. [0], because it is the first and only row)

    cache[labels] = p

    return p


def bestChild(root: MCTSNode) -> list[int]:  # best proba
    """
    Returns the labels of the best child of the root node following a greedy policy.

    Args:
        root (MCTSNode): The root node from which to select the best child
    """
    return select(root, eps=0).get_parent_labels()


def MCTS(model, x, verbose: bool = False, secs: float = 1) -> list[int]:
    """
    Monte Carlo Tree Search alogrithm.

    Args:
        model (Any): The model to use for the MCTS algorithm
        x (Any): The input data
        verbose (bool): If True, the constraints will print a message when they are reached
        secs (float): The time constraint in seconds

    Returns:
        list[int]: The labels of the best child of the root node following a greedy policy
    """
    n_classes: int = len(model.estimators_)
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=n_classes, proba=1)

    ComputationalConstraint: Constraint = Constraint(time=True, d_time=secs, max_iter=False, n_iter=0, verbose=verbose)

    cache: Dict[list[int], float] = {}  # Create a cache to store the reward evaluation to gain inference speed
    while (ComputationalConstraint):
        node: MCTSNode = select(root)
        reward: float = simulate(node, model, x, cache)
        back_prog(node, reward)

    return bestChild(root)


def func2() -> None:
    n_samples: int = 1000
    n_features: int = 6
    n_classes: int = 3
    n_labels: int = 2
    random_state: int = 0

    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=n_labels,
        random_state=random_state)

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    print(f"{X_train.shape = }\n{X_train}")
    print(f"{Y_train.shape = }\n{Y_train}")

    solver = "liblinear"
    base = LogisticRegression(solver=solver)
    chain: ClassifierChain = ClassifierChain(base)

    chain = chain.fit(X_train, Y_train)

    M = 10
    for i in range(M):
        print(f"MCTS Pred:{MCTS(chain, X_test[i])}, ClassifierChain Pred:{chain.predict(X_test[i].reshape(1,-1))[0]} vs True:{Y_test[i]}")

    def func3() -> None:
        from tqdm import trange

        secs = 0.1  # 0.5 Second per inference to test
        M = 100

        _y_mcts: list[list[int]] = []
        y_chain: np.ndarray[Any, np.dtype[Any]] = chain.predict(X_test[:M])

        for i in trange(M, desc=f"MCTS Inference Constraint={secs}s", unit="it", colour="green"):
            _y_mcts.append(MCTS(chain, X_test[i], secs=secs))

        y_mcts: np.ndarray[Any, np.dtype[Any]] = np.array(_y_mcts)

        from sklearn.metrics import hamming_loss, zero_one_loss

        print(f"MCTS  vs TRUE : Hamming={hamming_loss(y_mcts,Y_test[:M]):.4f}, ZeroOne={zero_one_loss(y_mcts,Y_test[:M]):.4f}")
        print(f"CHAIN vs TRUE : Hamming={hamming_loss(y_chain,Y_test[:M]):.4f}, ZeroOne={zero_one_loss(y_chain,Y_test[:M]):.4f}")
        print(f"MCTS  vs CHAIN: Hamming={hamming_loss(y_chain,y_mcts):.4f}, ZeroOne={zero_one_loss(y_chain,y_mcts):.4f}")


def main() -> None:
    print("Hello World!")
    func2()
    print("Done")


if __name__ == "__main__":
    main()
