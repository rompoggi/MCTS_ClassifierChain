from typing import Optional, Any, List, TypeVar
import numpy as np
from graphviz import Digraph

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
        get_parent_labels(self) -> list[int]: Get the labels of the parent nodes
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
        out: str = f"(MCTSNode: L={self.label}, R={self.rank}, P={self.score:.4f}, PL{self.parent_labels})"
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

def visualize_tree(root: MCTSNode, best_child: Optional[list[int]] = None, name: str = "binary_tree", save: bool = False, view: bool = True) -> None:
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


__all__: List[str] = ["MCTSNode", "visualize_tree", "normalize_score"]
