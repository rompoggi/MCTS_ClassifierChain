"""
Test file for the mcts module.
"""

import pytest
from mcts_inference.mcts import randmax


@pytest.mark.parametrize("arr, argmax", [([0, 0, 1], 2), ([0, 10, -1], 1)])
def test_randmax_unique_max(arr, argmax) -> None:
    assert (randmax(arr) == argmax)


@pytest.mark.parametrize("arr, argmaxs", [([0, 1, 1, 0, 0], [1, 2]), ([11, 10, -1, 11, 0], [0, 3])])
def test_randmax_mult_max(arr, argmaxs) -> None:
    assert (randmax(arr) in argmaxs)
