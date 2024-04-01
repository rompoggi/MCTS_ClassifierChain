"""
This module contains the implementation of the Monte Carlo Tree Search (MCTS) algorithm, along with the necessary node class MCTSConfig.
The MCTS algorithm is used to find the best child of a given node in a tree.
The tree is built by expanding the nodes and propagating the rewards back up the tree.

There are different policies to select the next node to visit, such as epsilon greedy here.
"""

import numpy as np
from typing import Any, Dict, Tuple, List, Optional, TypeVar
import multiprocessing as mp
from tqdm import tqdm
from graphviz import Digraph

from .constraints import Constraint
from .policy import Policy, Uniform, Greedy
from .utils import NormOption


T = TypeVar('T', bound='MCTSNode')  # Define the type now to not have an issue with recursive typing


class MCTSNode:
    """
    MCTSNode class to represent a node in the MCTS algorithm.

    Args:
        label (Optional[int]): The label of the node
        rank (int): The rank of the node
        n_children (int): The number of children for the node
        score (float): The probability of the node
        parent (Optional[MCTSNode]): The parent node
        parent_labels (List[int]): The labels of the parent nodes

    Attributes:
        score (float): The probability of the node
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
        get_parent_labels(self) -> List[int]: Get the labels of the parent nodes
        __str__(self) -> str: String representation of the node
        __repr__(self) -> str: String representation of the node
        print_all(self) -> None: Print all the attributes of the node in a readable format
        check_correct_count(self) -> bool: Checks recursively that all the children's visit count sum to that of their parent node
        is_fully_expanded(self) -> bool: Checks recursively if the entire tree has been expanded
        normalize_proba(self, opt: NormOption = NormOption.SOFTMAX) -> None: Normalizes the rewards obtained at each node into a distribution

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
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
        >>> node[0].parent
        (MCTSNode: L=0, R=2, P=0.5000, PL[])
    """
    def __init__(self,
                 label: Optional[int] = 0,
                 rank: int = 2,
                 n_children: int = 2,
                 score: float = 0.,
                 parent: Optional["MCTSNode"] = None,
                 parent_labels: List[int] = []) -> None:

        self.score: float = score
        self.label: Optional[int] = label
        self.visit_count: int = 0

        self.parent: Optional[MCTSNode] = parent
        self.children: List[MCTSNode] = []
        self.n_children: int = n_children
        self.rank: int = rank

        self.parent_labels: List[int] = parent_labels

    def __getitem__(self, key: int) -> "MCTSNode":
        """
        get the child node at the given key
        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
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

    def is_root(self) -> bool:
        """
        Returns True if the node is the root, False otherwise.
        A node is a root if it's label is None.
        """
        return (self.label is None)

    def expand(self) -> None:
        """
        Expand the node by creating its children.
        Each child will have a rank one less than the parent node.
        """
        assert (not self.is_terminal()), "Cannot expand a terminal node"
        assert (not self.is_expanded()), "Node already expanded. Cannot expand again."
        self.children = [MCTSNode(label=i, rank=self.rank-1, n_children=self.n_children,
                                  parent=self, parent_labels=self.parent_labels+[i]) for i in range(self.n_children)]

    def get_parent_labels(self) -> List[int]:
        """
        Get the labels of the parent nodes.
        """
        return self.parent_labels

    def __str__(self) -> str:
        """
        String representation of the node
        """
        out: str = f"(MCTSNode: L={self.label}, R={self.rank}, P={self.score:.4f}, PL{self.parent_labels})"
        return out

    def __repr__(self) -> str:
        """
        String representation of the node
        """
        return str(self)

    def __eq__(self, __value: object) -> bool:
        """
        Check if the two nodes are equal
        """
        if not isinstance(__value, MCTSNode):
            return False
        return (repr(self) == repr(__value))

    def print_all(self) -> None:
        """
        Print all the attributes of the node in a readable format.
        """
        print("\n".join([f"{k}:{v}" for k, v in self.__dict__.items()]))

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
        return self.is_expanded()

    def check_correct_count(self) -> bool:
        """
        Checks recursively that all the children's visit count sum to that of their parent node.

        Args:
            node (MCTSNode): The node from which to start the check

        Returns:
            bool: True if the visit count sum is correct, False otherwise

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node[0].visit_count = 2  # simulate that child 0 has been visited twice
            >>> node[1].visit_count = 3  # simulate that child 1 has been visited three times
            >>> node.visit_count
            0
            >>> node.check_correct_count()
            False
            >>> node.visit_count = 5
            >>> node.check_correct_count()
            True
        """
        assert (self.is_fully_expanded())

        def aux(node: MCTSNode) -> bool:
            count: int = 0
            if node.is_terminal():
                return True
            for child in node.children:
                if not aux(child):
                    return False
                count += child.visit_count
            return (count == node.visit_count)
        return aux(self)

    def get_children_scores(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Get the scores of the children nodes
        """
        assert (self.is_expanded()), "Node not yet expanded. Cannot get the children scores."
        return np.array([child.score for child in self.children], dtype=np.float64)

    def get_children_counts(self) -> np.ndarray[Any, np.dtype[np.int64]]:
        """
        Get the scores of the children nodes
        """
        assert (self.is_expanded()), "Node not yet expanded. Cannot get the children scores."
        return np.array([child.visit_count for child in self.children], dtype=np.int64)


###################################################################################################
# End of the MCTSNode class methods. We define other functions to be used with the MCTSNode class #
###################################################################################################
def visualize_tree(root: MCTSNode,
                   best_child: Optional[List[int]] = None,
                   name: str = "binary_tree",
                   save: bool = False,
                   view: bool = True) -> None:  # pragma: no cover
    """
    Visualize the search tree using the graphviz library.

    Args:
        root (MCTSNode): The root node of the tree
        with_path (bool): If True, the best path will be highlighted in red
        name (str): The name of the file to save the visualization

    Returns:
        None
    """
    assert (view or save), "At least one of view or save must be True"
    dot = Digraph()

    def add_nodes_edges(node: MCTSNode) -> None:
        dot.node(str(id(node)), label=f"{node.label}, {node.visit_count}")
        if node.children:
            for child in node.children:
                dot.edge(str(id(node)), str(id(child)), label=f"{child.score:.3f}")
                add_nodes_edges(child)

    add_nodes_edges(root)

    if best_child is not None:
        for i in range(len(best_child)):
            current_node_id = str(id(root))
            next_node_id = str(id(root[best_child[i]]))
            dot.edge(current_node_id, next_node_id, color="red")
            root = root[best_child[i]]

        dot.render(name + '_with_path', format='png', view=view, cleanup=not (save), quiet=True)
        return

    dot.render(name, format='png', view=view, cleanup=not (save), quiet=True)


def normalize_score(root: MCTSNode, opt: NormOption = NormOption.SOFTMAX) -> None:
    """
    Normalizes the scores obtained at each node into a distribution

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node[0].score = 0.
        >>> node[1].score = 2.
        >>> node.normalize_score(opt=NormOption.SOFTMAX)
        >>> node[0].score
        0.0
        >>> node[1].score
        1.0
    """
    assert (root.is_root())

    if opt == NormOption.SOFTMAX:
        _normalize_score_softmax(root)
    elif opt == NormOption.UNIFORM:
        _normalize_score_uniform(root)
    elif opt == NormOption.NONE:
        _normalize_score_none(root)


def _normalize_score_softmax(node: MCTSNode) -> None:
    if node.is_terminal():
        return
    scores: np.ndarray[Any, np.dtype[np.float64]] = node.get_children_scores()
    scores = np.exp(scores)
    scores /= np.sum(scores)

    for i, child in enumerate(node.children):
        _normalize_score_softmax(child)
        child.score = scores[i]


def _normalize_score_uniform(node: MCTSNode) -> None:
    if node.is_terminal():
        return
    scores: np.ndarray[Any, np.dtype[np.float64]] = node.get_children_scores()
    assert (min(scores) > 0)
    scores /= np.sum(scores)

    for i, child in enumerate(node.children):
        _normalize_score_uniform(child)
        child.score = scores[i]


def _normalize_score_none(node: MCTSNode) -> None:
    pass  # Do nothing

##################################################
#               MCTS Algorithm                   #
##################################################


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


def MCTS(x, model, config) -> Any:  # pragma: no cover
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


__all__: list[str] = ["MCTS", "MCTSNode", "visualize_tree", "normalize_score"]
