"""
Test file for the mcts_node module.
"""

import pytest
import numpy as np
from typing import List, Tuple

from mcts_inference.mcts_node import MCTSNode, normalize_score, visualize_tree
from mcts_inference.utils import NormOption


######################################################################
#                             get_item                               #
######################################################################
def test_getitem() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    with pytest.raises(AssertionError):
        node[0]

    node.expand()
    assert (node[0].label == 0)
    assert (node[1].label == 1)

    with pytest.raises(AssertionError):
        node[-1]
    with pytest.raises(AssertionError):
        node[2]


######################################################################
#                           is_terminal                              #
######################################################################
def test_is_terminal() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    assert (node.is_terminal() is False)
    node.expand()
    assert (node.is_terminal() is False)
    node[0].expand()
    assert (node[0][0].is_terminal() is True)


######################################################################
#                           is_expanded                              #
######################################################################
def test_is_expanded_unexpanded_node() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    assert node.is_expanded() is False, "is_expanded should return False for an unexpanded node"


def test_is_expanded_expanded_node() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    assert node.is_expanded() is True, "is_expanded should return True for an expanded node"


######################################################################
#                             is_root                                #
######################################################################
def test_is_root_root_node() -> None:
    node = MCTSNode(label=None, rank=2, n_children=2)
    assert node.is_root() is True, "is_root should return True for a root node"


def test_is_root_non_root_node() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    assert node.is_root() is False, "is_root should return False for a non-root node"


######################################################################
#                              expand                                #
######################################################################
def test_expand_terminal_node() -> None:
    node = MCTSNode(label=0, rank=0, n_children=2)
    with pytest.raises(AssertionError):
        node.expand()


def test_expand_already_expanded_node() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    with pytest.raises(AssertionError):
        node.expand()


def test_expand_valid_node() -> None:
    node = MCTSNode(label=None, rank=2, n_children=2)
    node.expand()
    assert len(node.children) == 2, "expand should create 2 children for the node"
    assert all(child.rank == 1 for child in node.children), "All children should have rank 1"
    assert all(child.parent == node for child in node.children), "All children should have the original node as parent"
    node[0].expand()
    assert all(child.parent_labels == [0, child.label] for child in node[0].children), "All children should have [0] as parent labels"


######################################################################
#                        get_parent_labels                           #
######################################################################
def test_get_parent_labels() -> None:
    node = MCTSNode(label=None, rank=2, n_children=2)
    node.expand()
    assert (node.get_parent_labels() == [])
    assert (node[0].get_parent_labels() == [0])
    node[0].expand()
    assert (node[0][1].get_parent_labels() == [0, 1])
    node.parent_labels = [1, 2, 4]
    assert (node.get_parent_labels() == [1, 2, 4])


######################################################################
#                              __str__                               #
######################################################################
ins_str: List[Tuple[int, int, float, List[int | None]]] = [
    (0, 1, 0.1, [1, 0]),
    (12, 324, 12., [None])
]


@pytest.mark.parametrize("label, rank, score, parent_labels", ins_str)
def test_str(label, rank, score, parent_labels) -> None:
    node = MCTSNode(label=label, rank=rank, score=score, parent_labels=parent_labels)
    assert (str(node) == f"(MCTSNode: L={label}, R={rank}, P={score:.4f}, PL{parent_labels})")


######################################################################
#                              __repr__                              #
######################################################################
ins_repr: List[Tuple[int, int, float, List[int | None]]] = [
    (0, 1, 0.1, [1, 0]),
    (12, 324, 12., [None])
]


@pytest.mark.parametrize("label, rank, score, parent_labels", ins_repr)
def test_repr(label, rank, score, parent_labels) -> None:
    node = MCTSNode(label=label, rank=rank, score=score, parent_labels=parent_labels)
    assert (repr(node) == f"(MCTSNode: L={label}, R={rank}, P={score:.4f}, PL{parent_labels})")


######################################################################
#                               __eq__                               #
######################################################################
def test_eq_same_node() -> None:
    node1 = MCTSNode(label=0, rank=1, score=0., n_children=2)
    assert node1 == node1, "A node should be equal to itself"


def test_eq_identical_nodes() -> None:
    node1 = MCTSNode(label=0, rank=1, score=0., n_children=2)
    node2 = MCTSNode(label=0, rank=1, score=0., n_children=2)
    assert node1 == node2, "Two nodes with the same attributes should be equal"


def test_eq_different_nodes() -> None:
    node1 = MCTSNode(label=0, rank=1, score=0., n_children=2)
    node2 = MCTSNode(label=1, rank=1, score=0., n_children=2)
    assert node1 != node2, "Two nodes with different attributes should not be equal"


def test_eq_non_node() -> None:
    node = MCTSNode(label=0, rank=1, score=0., n_children=2)
    assert node != "not a node", "A node should not be equal to a non-node object"


######################################################################
#                            print_all                               #
######################################################################
def test_print_all(capfd) -> None:
    node = MCTSNode(label=0, rank=1, score=0., n_children=2)
    node.print_all()
    captured = capfd.readouterr()
    assert (captured.out == "\n".join([f"{k}:{v}" for k, v in node.__dict__.items()])+"\n")


######################################################################
#                        is_fully_expanded                           #
######################################################################
def test_is_fully_expanded() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    node[0].expand()

    assert (not node.is_fully_expanded())

    node[1].expand()
    assert (node.is_fully_expanded())


######################################################################
#                       check_correct_count                          #
######################################################################
def test_check_correct_count() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    node[0].expand()
    node[1].expand()
    assert (node.check_correct_count())

    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    node[0].expand()
    node[1].expand()

    node[0][0].visit_count = 2
    node[0][1].visit_count = 3
    assert (not node.check_correct_count())
    node[0].visit_count = 5
    assert (not node.check_correct_count())
    node.visit_count = 5
    assert (node.check_correct_count())


######################################################################
#                       get_children_scores                          #
######################################################################
def test_get_children_scores() -> None:
    node = MCTSNode(label=0, rank=2, n_children=3)
    with pytest.raises(AssertionError):
        node.get_children_scores()
    node.expand()
    assert (node.get_children_scores() == np.zeros_like(node.get_children_scores())).all

    node[0].score = 1.
    assert (len(node.get_children_scores()) == node.n_children)
    assert (node.get_children_scores() == np.array([1, 0., 0.])).all


######################################################################
#                       get_children_counts                          #
######################################################################
def test_get_children_counts() -> None:
    node = MCTSNode(label=0, rank=2, n_children=3)
    with pytest.raises(AssertionError):
        node.get_children_counts()
    node.expand()
    assert (node.get_children_counts() == np.zeros_like(node.get_children_counts())).all

    node[0].visit_count = 10
    assert (len(node.get_children_counts()) == node.n_children)
    assert (node.get_children_counts() == np.array([10, 0, 0.])).all


######################################################################
#                       normalize_score SOFTMAX                      #
######################################################################
def test_normalize_score_softmax() -> None:
    node = MCTSNode(label=None, rank=1, n_children=2)
    node.expand()
    node[0].score = 1.
    node[1].score = 2.
    normalize_score(node, opt=NormOption.SOFTMAX)
    scores = np.exp([1., 2.])
    scores /= np.sum(scores)
    assert np.allclose(node[0].score, scores[0])
    assert np.allclose(node[1].score, scores[1])


######################################################################
#                       normalize_score UNIFORM                      #
######################################################################
def test_normalize_score_uniform() -> None:
    node = MCTSNode(label=None, rank=1, n_children=2)
    node.expand()
    node[0].score = 1.
    node[1].score = 2.
    normalize_score(node, opt=NormOption.UNIFORM)
    scores = np.array([1., 2.])
    scores /= np.sum(scores)
    assert np.allclose(node.children[0].score, scores[0])
    assert np.allclose(node.children[1].score, scores[1])


######################################################################
#                       normalize_score NONE                         #
######################################################################
def test_normalize_score_none() -> None:
    node = MCTSNode(label=None, rank=2, n_children=2)
    node.expand()
    node.children[0].score = 1.
    node.children[1].score = 2.
    normalize_score(node, opt=NormOption.NONE)
    assert np.allclose(node.children[0].score, 1.)
    assert np.allclose(node.children[1].score, 2.)
    node[0].expand()
    node[0][0].score = 1.
    with pytest.raises(AssertionError):
        normalize_score(node[0][0], opt=NormOption.NONE)
    node[0][0].label = None
    normalize_score(node[0][0], opt=NormOption.NONE)
    assert (node[0][0].score == 1.)


######################################################################
#                          visualize_tree                            #
######################################################################
# @pytest.mark.skip(reason="Bug with graphviz")
def test_visualize_tree() -> None:
    root = MCTSNode(label=None, rank=2, n_children=2)
    with pytest.raises(AssertionError):
        visualize_tree(root, view=False, save=False)
    root.expand()
    root[0].expand()
    root[1].expand()
    visualize_tree(root, None, view=False, save=True)
    visualize_tree(root, [0, 1], view=False, save=True)
    import os

    assert (os.path.exists("binary_tree.png"))
    os.remove("binary_tree.png")
    os.remove("binary_tree")
    assert (not os.path.exists("binary_tree.png")), "A problem occured during cleanup"
    assert (not os.path.exists("binary_tree")), "A problem occured during cleanup"

    assert (os.path.exists("binary_tree_with_path.png"))
    os.remove("binary_tree_with_path.png")
    os.remove("binary_tree_with_path")
    assert (not os.path.exists("binary_tree_with_path.png")), "A problem occured during cleanup"
    assert (not os.path.exists("binary_tree_with_path")), "A problem occured during cleanup"
