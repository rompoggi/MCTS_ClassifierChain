"""
Test file for the mcts module.
"""

from unittest import mock
import numpy as np
import pytest
from typing import List, Tuple, Dict, Any

from mcts_inference.mcts import MCTSNode, normalize_score, visualize_tree
from mcts_inference.mcts import select, simulate, get_reward, back_prog
from mcts_inference.mcts import best_child, best_child_all
from mcts_inference.mcts import MCTS_all_step_atime_wrapper, MCTS_one_step_atime_wrapper

from mcts_inference.policy import Policy, Uniform, Greedy

from mcts_inference.utils import NormOption


###########################################################################
#                                                                         #
#                            Test for MCTSNode classes                    #
#                                                                         #
###########################################################################

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
@pytest.mark.skip(reason="Bug with graphviz")
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

###########################################################################
#                                                                         #
#                              Test for MCTS Algorithm                    #
#                                                                         #
###########################################################################


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
    assert (node.visit_count == v_count)
    assert (node[0] == result or node[1] == result)


def test_select_terminal() -> None:
    policy: Policy = Greedy()

    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)

    assert (not node.is_terminal())
    node.expand()

    result: MCTSNode = select(node, policy)
    assert ((result == node[0]) or (result == node[1]))


######################################################################
#                           back_prog                                #
######################################################################
def test_back_prog_root_node() -> None:
    root: MCTSNode = MCTSNode(label=0, n_children=2, rank=1, score=0.)
    back_prog(root, 1.0)
    assert (root.score == 0.), "back_prog should not update the score of the root node"
    assert (root.visit_count == 1), "back_prog should update the visit count of the root node"


@pytest.mark.parametrize("initial_score, reward", [(0., 1.), (1., 42.)])
def test_back_prog_non_root_node(initial_score, reward) -> None:
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=1, score=0.)
    root.expand()
    back_prog(root[0], initial_score)
    assert (root.score == 0.), "back_prog should not update the score of the root node"
    assert (root.visit_count == 1), "back_prog should update the visit count of the root node"
    assert (root[0].score == initial_score), "back_prog should update the score of the non-root node"
    assert (root[0].visit_count == 1), "back_prog should update the visit count of the non-root node"
    back_prog(root[0], reward)
    back_prog(root[1], reward)
    assert (root[0].score == (initial_score + reward)/2)
    assert (root[1].score == reward)
    assert (root.visit_count == 3), "back_prog should update the visit count of the root node"


ins_back_prog_recursive: List[Tuple[List[int], float]] = [([0, 0, 1], 42.), ([0, 1, 0, 1, 0, 0], 42.)]


@pytest.mark.parametrize("path, reward", ins_back_prog_recursive)
def test_back_prog_recursive(path, reward) -> None:
    n_children: int = 2
    root: MCTSNode = MCTSNode(label=None, n_children=2, rank=len(path))
    node: MCTSNode = root
    for p in path:  # Simulate selection for the path
        node.expand()
        node = node[p]

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
    assert (node.visit_count == initial_visit_count), "simulate should not increment the visit count"


def test_simulate_unexpanded_node() -> None:
    policy: Policy = Uniform()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)
    initial_visit_count: int = node.visit_count
    result: MCTSNode = simulate(node, policy)
    assert (node != result), "simulate should return a different node if the initial node is not terminal"
    assert (node.is_expanded()), "simulate should expand the initial node if it is not terminal"
    assert (node.visit_count == initial_visit_count), "simulate should not increment the visit count"


def test_simulate_expanded_node() -> None:
    policy: Policy = Uniform()
    node: MCTSNode = MCTSNode(label=0, n_children=2, rank=1)
    node.expand()
    initial_visit_count: int = node.visit_count
    result: MCTSNode = simulate(node, policy)
    assert (node != result), "simulate should return a different node if the initial node is not terminal"
    assert (node.visit_count == initial_visit_count), "simulate should not increment the visit count"


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


def test_best_child_all() -> None:
    policy: Policy = Greedy()
    root: MCTSNode = MCTSNode(label=0, n_children=2, rank=2, score=0.)
    root.expand()
    root[0].score = 1.0
    root[1].score = 0.0
    assert (tuple(best_child_all(root, policy)) in {(0, 1), (0, 0)})


######################################################################
#                     MCTS_all_step_atime                         #
######################################################################
def test_MCTS_all_step_atime() -> None:
    # Mock the _all_step_MCTS function
    with mock.patch('mcts_inference.mcts.MCTS_all_step_atime', return_value=[1, 2, 3]) as mock_MCTS_all_step_atime:
        result = MCTS_all_step_atime_wrapper((1, 'a', True))
        mock_MCTS_all_step_atime.assert_called_once_with(1, 'a', True)
        assert result == [1, 2, 3], "MCTS_all_step_atime_wrapper should return the same result as MCTS_all_step_atime"

    with mock.patch('mcts_inference.mcts.MCTS_all_step_atime', return_value=[4, 5, 6]) as mock_MCTS_all_step_atime:
        result = MCTS_all_step_atime_wrapper(('test', 2.5, False))
        mock_MCTS_all_step_atime.assert_called_once_with('test', 2.5, False)
        assert result == [4, 5, 6], "MCTS_all_step_atime_wrapper should return the same result as MCTS_all_step_atime"


######################################################################
#                     MCTS_one_step_atime                         #
######################################################################
def test_MCTS_one_step_atime() -> None:
    # Mock the MCTS_one_step_atime function
    with mock.patch('mcts_inference.mcts.MCTS_one_step_atime', return_value=[1, 2, 3]) as mock_MCTS_one_step_atime:
        result = MCTS_one_step_atime_wrapper((1, 'a', True))
        mock_MCTS_one_step_atime.assert_called_once_with(1, 'a', True)
        assert result == [1, 2, 3], "MCTS_one_step_atime_wrapper should return the same result as MCTS_one_step_atime"

    with mock.patch('mcts_inference.mcts.MCTS_one_step_atime', return_value=[4, 5, 6]) as mock_MCTS_one_step_atime:
        result = MCTS_one_step_atime_wrapper(('test', 2.5, False))
        mock_MCTS_one_step_atime.assert_called_once_with('test', 2.5, False)
        assert result == [4, 5, 6], "MCTS_one_step_atime_wrapper should return the same result as MCTS_one_step_atime"
