import pytest
from unittest import mock

import numpy as np

from mcts_inference.mc import MCNode, MC_wrapper


def test_MCNode() -> None:
    node: MCNode = MCNode(label=1, rank=2, n_children=2, score=0.5)
    assert (node.label == 1)
    assert (node.rank == 2)
    assert (node.n_children == 2)
    assert (node.score == 0.5)
    assert (node.is_terminal() is False)
    assert (node.is_expanded() is False)


def test_get_children_scores() -> None:
    node = MCNode(label=1, rank=2, n_children=2, score=0.5)
    with pytest.raises(AssertionError):
        node.get_children_scores()
    node.children = [MCNode(label=1, rank=2, n_children=2, score=0.5), MCNode(label=1, rank=2, n_children=2, score=0.5)]
    scores = node.get_children_scores()
    assert (scores[0] == 0.5)
    assert (scores[1] == 0.5)
    node[0].score = 0.6
    assert ((node.get_children_scores() == np.array([0.6, 0.5])).all())


def test_get_parent_labels() -> None:
    node = MCNode(label=None, rank=2, n_children=2, score=0.5)
    assert (node.get_parent_labels() == [])
    node = MCNode(label=1, rank=2, n_children=2, score=0.5, parent_labels=[1, 2])
    assert (node.get_parent_labels() == [1, 2])


def test_is_fully_expanded() -> None:
    node = MCNode(label=1, rank=2, n_children=2, score=0.5)
    assert (node.is_fully_expanded() is False)
    node.children = [MCNode(label=0, rank=1, n_children=2, score=0.5), MCNode(label=1, rank=1, n_children=2, score=0.5)]
    assert (node.is_fully_expanded() is False)
    node[0].children = [MCNode(label=0, rank=0, n_children=2, score=0.5), MCNode(label=1, rank=0, n_children=2, score=0.5)]
    assert (node.is_fully_expanded() is False)
    node[1].children = [MCNode(label=0, rank=0, n_children=2, score=0.5), MCNode(label=1, rank=0, n_children=2, score=0.5)]
    assert (node.is_fully_expanded() is True)


def test_MC_wrapper() -> None:
    with mock.patch('mcts_inference.mc._MC', return_value=[1, 2, 3]) as mock_MC:
        result = MC_wrapper((1, 'a', True))
        mock_MC.assert_called_once_with(1, 'a', True)
        assert result == [1, 2, 3], "MC_wrapper should return the same result as _MC"

    with mock.patch('mcts_inference.mc._MC', return_value=[4, 5, 6]) as mock_MC:
        result = MC_wrapper(('test', 2.5, False))
        mock_MC.assert_called_once_with('test', 2.5, False)
        assert result == [4, 5, 6], "MC_wrapper should return the same result as _MC"
