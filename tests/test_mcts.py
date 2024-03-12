"""
Test file for the mcts module.
"""

from unittest import mock
import numpy as np
import pytest
from typing import List, Tuple, Dict, Any

from mcts_inference.mcts import select, back_prog, simulate, get_reward, best_child, best_child_all, MCTS_all_step_atime, MCTS_one_step_atime
from mcts_inference.mcts_node import MCTSNode
from mcts_inference.policy import Policy, Uniform, Greedy


######################################################################
#                             select                                 #
######################################################################
def test_select_expanded() -> None:
    policy: Policy = Uniform()

    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)
    v_count: int = node.visit_count

    result: MCTSNode = select(node, policy)
    assert (node == result)
    assert (node.visit_count == v_count)

    node.expand()

    result = select(node, policy)
    assert (node != result)
    assert (node.visit_count == v_count + 1)
    assert (node[0] == result or node[1] == result)


def test_select_terminal() -> None:
    policy: Policy = Greedy()

    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=0)

    result: MCTSNode = select(node, policy)
    assert (node == result)


######################################################################
#                           back_prog                                #
######################################################################
def test_back_prog_root_node() -> None:
    root: MCTSNode = MCTSNode(label=0, n_children=2, rank=1, score=0.)
    back_prog(root, 1.0)
    assert (root.score == 0.), "back_prog should not update the score of the root node"


@pytest.mark.parametrize("initial_score, reward", [(0., 1.), (1., 42.)])
def test_back_prog_non_root_node(initial_score, reward) -> None:
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=1, score=0.)
    root.expand()
    with pytest.raises(AssertionError):
        back_prog(root[0], 1.0)
    root[0].visit_count = 2  # node visited once for the initial score we give it, and once more of the back_prog
    root[0].score = initial_score
    root.visit_count += 1
    back_prog(root[0], reward)
    assert (root.score == 0.)
    assert (root[0].score == (initial_score + reward)/2)


ins_back_prog_recursive: List[Tuple[List[int], float]] = [([0, 0, 1], 42.), ([0, 1, 0, 1, 0, 0], 42.)]


@pytest.mark.parametrize("path, reward", ins_back_prog_recursive)
def test_back_prog_recursive(path, reward) -> None:
    n_children: int = 2
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=len(path))
    node: MCTSNode = root
    for p in path:  # Simulate selection for the path
        node.expand()
        node.visit_count += 1
        node = node[p]
    with pytest.raises(AssertionError):  # visit count is 0
        back_prog(node, reward)
    node.visit_count += 1

    back_prog(node, reward)

    assert all(root.get_children_scores() == [reward * int(path[0] == i) for i in range(n_children)])

    node = root
    for p in path:
        assert all(node.get_children_scores() == [reward*int(p == i) for i in range(n_children)])
        node = node[p]


######################################################################
#                            simulate                                #
######################################################################
def test_simulate_terminal_node() -> None:
    policy: Policy = Uniform()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=0)
    initial_visit_count: int = node.visit_count
    result: MCTSNode = simulate(node, policy)
    assert (node == result), "simulate should return the same node if it is terminal"
    assert (node.visit_count == initial_visit_count + 1), "simulate should increment the visit count of the terminal node"


def test_simulate_unexpanded_node() -> None:
    policy: Policy = Uniform()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)
    initial_visit_count: int = node.visit_count
    result: MCTSNode = simulate(node, policy)
    assert (node != result), "simulate should return a different node if the initial node is not terminal"
    assert (node.is_expanded()), "simulate should expand the initial node if it is not terminal"
    assert (node.visit_count == initial_visit_count + 1), "simulate should increment the visit count of the initial node"


def test_simulate_expanded_node() -> None:
    policy: Policy = Uniform()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)
    node.expand()
    initial_visit_count: int = node.visit_count
    result: MCTSNode = simulate(node, policy)
    assert (node != result), "simulate should return a different node if the initial node is not terminal"
    assert (node.visit_count) == initial_visit_count + 1, "simulate should increment the visit count of the initial node"


######################################################################
#                           get_reward                               #
######################################################################
class DummyEstimator:
    def __init__(self, val: float, n: int = 2) -> None:
        self.val: float = val
        self.n: int = n

    def predict_proba(self, _) -> List[List[float]]:
        return [[self.val] * self.n]


class DummyModel:
    def __init__(self, n_label: int = 2, val: float = 0.5, n: int = 2) -> None:
        self.estimators_: List[DummyEstimator | None] = [DummyEstimator(val=val, n=n)] * n_label


@pytest.fixture
def root() -> MCTSNode:
    root = MCTSNode(label=None, n_children=2, rank=2)
    root.expand()
    root[0].expand()
    root[1].expand()
    root.visit_count += 1
    root[0].visit_count += 1
    root[1].visit_count += 1
    return root


def test_get_reward_terminal_node(root) -> None:
    val1: float = 0.5
    val2: float = 1.
    model1 = DummyModel(n_label=root.rank, val=val1)
    x: np.ndarray[Any, np.dtype[int]] = np.array([1, 2, 3])
    cache: Dict[Tuple[int, ...], float] = {}
    node: MCTSNode = root[0][0]
    reward: float = get_reward(node, x, model1, cache)
    assert (reward == val1 ** root.rank), "get_reward should return the correct reward for a terminal node"
    assert (cache == {(0, 0): val1 ** root.rank}), "get_reward should update the cache correctly"

    model2 = DummyModel(n_label=root.rank, val=val2)
    reward = get_reward(node, x, model2, cache)
    assert (reward == val1 ** root.rank)  # same reward because of cache.

    node = root[0][1]
    reward = get_reward(node, x, model2, cache)
    assert (reward == val2 ** root.rank), "get_reward should return the correct reward for a terminal node"
    assert (cache == {(0, 0): val1 ** root.rank, (0, 1): val2 ** root.rank})

    node = root[1]
    node.parent_labels = []
    node[0].parent_labels = [1]
    node.parent = None
    reward = get_reward(node[0], x=x, model=model2, cache=cache, ys=[1])
    assert (reward == val2 ** root.rank)


def test_has_predict_proba(root) -> None:
    model = DummyModel(2, 0.5, 2)
    model.estimators_[0] = None
    with pytest.raises(AssertionError):
        get_reward(node=root[0][0], x=np.array([1, 2, 3]), model=model, cache={})


######################################################################
#                           best_child                               #
######################################################################
def test_best_child_root_node() -> None:
    policy: Policy = Greedy()
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=1, score=0.)
    root.expand()
    root.visit_count += 1
    root[0].visit_count += 1
    root[1].visit_count += 1
    root[0].score = 1.0
    root[1].score = 2.0
    assert (best_child(root, policy) == 1), "best_child should return the labels of the child with the highest score"
    assert (best_child_all(root, policy) == [1]), "best_child should return the labels of the child with the highest score"


def test_best_child_non_root_node() -> None:
    policy: Policy = Greedy()
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=2, score=0.)
    root.expand()
    root.visit_count += 1
    root[0].visit_count += 1
    root[1].visit_count += 1
    node: MCTSNode = root[0]
    node.expand()
    node[0].score = 1.0
    node[1].score = 2.0
    assert (best_child_all(node, policy) == [0, 1]), "best_child should return the labels of the child with the highest score"


def test_best_child_no_children() -> None:
    policy: Policy = Greedy()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1, score=0.)
    with pytest.raises(AssertionError):
        best_child(node, policy)


######################################################################
#                     MCTS_all_step_atime                         #
######################################################################
def testMCTS_all_step_atime() -> None:
    # Mock the _all_step_MCTS function
    with mock.patch('mcts_inference.mcts._all_step_MCTS', return_value=[1, 2, 3]) as mock_all_step_MCTS:
        result = MCTS_all_step_atime((1, 'a', True))
        mock_all_step_MCTS.assert_called_once_with(1, 'a', True)
        assert result == [1, 2, 3], "MCTS_all_step_atime should return the same result as _all_step_MCTS"

    with mock.patch('mcts_inference.mcts._all_step_MCTS', return_value=[4, 5, 6]) as mock_all_step_MCTS:
        result = MCTS_all_step_atime(('test', 2.5, False))
        mock_all_step_MCTS.assert_called_once_with('test', 2.5, False)
        assert result == [4, 5, 6], "MCTS_all_step_atime should return the same result as _all_step_MCTS"


######################################################################
#                     MCTS_one_step_atime                         #
######################################################################
def testMCTS_one_step_atime() -> None:
    # Mock the _one_step_MCTS function
    with mock.patch('mcts_inference.mcts._one_step_MCTS', return_value=[1, 2, 3]) as mock_one_step_MCTS:
        result = MCTS_one_step_atime((1, 'a', True))
        mock_one_step_MCTS.assert_called_once_with(1, 'a', True)
        assert result == [1, 2, 3], "MCTS_one_step_atime should return the same result as _one_step_MCTS"

    with mock.patch('mcts_inference.mcts._one_step_MCTS', return_value=[4, 5, 6]) as mock_one_step_MCTS:
        result = MCTS_one_step_atime(('test', 2.5, False))
        mock_one_step_MCTS.assert_called_once_with('test', 2.5, False)
        assert result == [4, 5, 6], "MCTS_one_step_atime should return the same result as _one_step_MCTS"
