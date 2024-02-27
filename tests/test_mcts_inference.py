"""
Test file for the mcts module.
"""

import numpy as np
import pytest
from mcts_inference.mcts import MCTSNode, randmax, normalize_score
from mcts_inference.utils import NormOption


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


def test_is_terminal() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    assert (node.is_terminal() is False)
    node.expand()
    assert (node.is_terminal() is False)
    node[0].expand()
    assert (node[0][0].is_terminal() is True)


def test_get_parent_labels() -> None:
    node = MCTSNode(label=None, rank=2, n_children=2)
    node.expand()
    assert (node.get_parent_labels() == [])
    assert (node[0].get_parent_labels() == [0])
    node[0].expand()
    assert (node[0][1].get_parent_labels() == [0, 1])
    node.parent_labels = [1, 2, 4]
    assert (node.get_parent_labels() == [1, 2, 4])


ins = [
    (0, 1, 0.1, [1, 0]),
    (12, 324, 12., [None])
]


@pytest.mark.parametrize("label, rank, score, parent_labels", ins)
def test_str(label, rank, score, parent_labels) -> None:
    node = MCTSNode(label=label, rank=rank, score=score, parent_labels=parent_labels)
    assert (str(node) == f"(MCTSNode: L={label}, R={rank}, P={score:.4f}, PL{parent_labels})")


@pytest.mark.parametrize("label, rank, score, parent_labels", ins)
def test_repr(label, rank, score, parent_labels) -> None:
    node = MCTSNode(label=label, rank=rank, score=score, parent_labels=parent_labels)
    assert (repr(node) == f"(MCTSNode: L={label}, R={rank}, P={score:.4f}, PL{parent_labels})")


def test_print_all(capfd) -> None:
    node = MCTSNode(label=0, rank=1, score=0., n_children=2)
    node.print_all()
    captured = capfd.readouterr()
    assert (captured.out == "\n".join([f"{k}:{v}" for k, v in node.__dict__.items()])+"\n")


def test_is_fully_expanded() -> None:
    node = MCTSNode(label=0, rank=2, n_children=2)
    node.expand()
    node[0].expand()

    assert (not node.is_fully_expanded())

    node[1].expand()
    assert (node.is_fully_expanded())


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


def test_get_children_scores() -> None:
    node = MCTSNode(label=0, rank=2, n_children=3)
    with pytest.raises(AssertionError):
        node.get_children_scores()
    node.expand()
    assert (node.get_children_scores() == np.zeros_like(node.get_children_scores())).all

    node[0].score = 1.
    assert (len(node.get_children_scores()) == node.n_children)
    assert (node.get_children_scores() == np.array([1, 0., 0.])).all


def test_get_children_counts() -> None:
    node = MCTSNode(label=0, rank=2, n_children=3)
    with pytest.raises(AssertionError):
        node.get_children_counts()
    node.expand()
    assert (node.get_children_counts() == np.zeros_like(node.get_children_counts())).all

    node[0].visit_count = 10
    assert (len(node.get_children_counts()) == node.n_children)
    assert (node.get_children_counts() == np.array([10, 0, 0.])).all


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


@pytest.mark.parametrize("arr, argmax", [([0, 0, 1], 2), ([0, 10, -1], 1)])
def test_randmax_unique_max(arr, argmax) -> None:
    assert (randmax(arr) == argmax)


@pytest.mark.parametrize("arr, argmaxs", [([0, 1, 1, 0, 0], [1, 2]), ([11, 10, -1, 11, 0], [0, 3])])
def test_randmax_mult_max(arr, argmaxs) -> None:
    assert (randmax(arr) in argmaxs)
