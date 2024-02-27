"""
Test file for the policy module.
"""

import pytest
from mcts_inference.mcts import MCTSNode
from mcts_inference.policy import Policy, Uniform, Greedy, EpsGreedy, UCB, Thompson_Sampling
import numpy as np


######################################################################
#                                                                    #
#                   Generic tests for Policy class                   #
#                                                                    #
######################################################################

policies_name = [
    (Uniform, {}, "Uniform"),
    (EpsGreedy, {'epsilon': 0.1}, "EpsGreedy"),
    (Greedy, {}, "Greedy"),
    (UCB, {'alpha': 0.5}, "UCB"),
    (Thompson_Sampling, {'a': 1., 'b': 1.}, "Thompson_Sampling")
]


@pytest.mark.parametrize("Pol, args, name", policies_name)
def test_policy_name(Pol, args, name) -> None:
    policy: Policy = Pol(**args)
    # Continue with your test
    assert (policy.name() == name)


policies_string = [
    (Uniform, {}, "Uniform"),
    (Greedy, {}, "Greedy"),
    (EpsGreedy, {'epsilon': 0.1}, f"EpsGreedy(epsilon={0.1})"),
    (UCB, {'alpha': 0.5}, f"UCB(alpha={0.5})"),
    (Thompson_Sampling, {'a': 1., 'b': 1.}, f"Thompson_Sampling(a={1.}, b={1.})")
]


@pytest.mark.parametrize("Pol, args, string", policies_string)
def test_policy_string(Pol, args, string) -> None:
    policy: Policy = Pol(**args)
    # Continue with your test
    assert (str(policy) == string)


policies_repr = [
    (Uniform, {}, "Uniform"),
    (Greedy, {}, "Greedy"),
    (EpsGreedy, {'epsilon': 0.1}, f"EpsGreedy(epsilon={0.1})"),
    (UCB, {'alpha': 0.5}, f"UCB(alpha={0.5})"),
    (Thompson_Sampling, {'a': 1., 'b': 1.}, f"Thompson_Sampling(a={1.}, b={1.})")
]


@pytest.mark.parametrize("Pol, args, rep", policies_repr)
def test_policy_repr(Pol, args, rep) -> None:
    policy: Policy = Pol(**args)
    # Continue with your test
    assert (repr(policy) == rep)


policies = [
    (Uniform, {}),
    (Greedy, {}),
    (EpsGreedy, {'epsilon': 0.1}),
    (UCB, {'alpha': 0.5}),
    (Thompson_Sampling, {'a': 1., 'b': 1.})
]


@pytest.mark.parametrize("Pol, args", policies)
def test_select_action(Pol, args) -> None:
    node = MCTSNode(label=0, rank=2, n_children=4)
    node.expand()
    node.visit_count = 1
    pol = Pol(**args)
    action = pol(node)
    assert isinstance(action, int), "The selected action should be an integer."


######################################################################
#                                                                    #
#                   Specific tests for Policy class                  #
#                                                                    #
######################################################################


######################################################################
#                               Policy                               #
######################################################################
def test_policy() -> None:
    policy = Policy()
    with pytest.raises(NotImplementedError):
        policy.name()
    with pytest.raises(NotImplementedError):
        policy.__call__(MCTSNode(label=0, rank=2, n_children=4))
    with pytest.raises(NotImplementedError):
        policy.__str__()
    with pytest.raises(NotImplementedError):
        policy.__repr__()


######################################################################
#                               Uniform                              #
######################################################################
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_Uniform(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=k)
    node.expand()
    uniform = Uniform()
    assert (uniform(node) in set(range(k)))


######################################################################
#                               Greedy                               #
######################################################################
@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_Greedy_set_score(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=k)
    node.expand()
    node.children[k-1].score = 1.0
    greedy = Greedy()
    assert (greedy(node) == k-1)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_Greedy_no_score(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=k)
    node.expand()
    greedy = Greedy()
    assert (greedy(node) in set(range(k)))


######################################################################
#                             EpsGreedy                              #
######################################################################
@pytest.mark.parametrize("epsilon", [-0.1, 2., 100.])
def test_EpsGreedy_invalid_init(epsilon) -> None:
    with pytest.raises(AssertionError):
        EpsGreedy(epsilon)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_EpsGreedy_esp1(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=k)
    node.expand()
    eps = EpsGreedy(epsilon=1.0)
    assert (eps(node) in set(range(k)))


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_EpsGreedy_eps0(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    node.children[k-1].score = 1.0
    eps = EpsGreedy(epsilon=0.0)
    assert (eps(node) == k-1)


######################################################################
#                                UCB                                 #
######################################################################
def test_ucb_visited() -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    ucb = UCB(alpha=0.5)
    with pytest.raises(AssertionError):
        ucb(node)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_ucb_no_visit(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    ucb = UCB(alpha=0.5)
    for i in range(node.n_children):
        if i != k-1:
            node.children[i].visit_count = 1
    node.visit_count = 7
    assert (ucb(node) == (k-1))


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_ucb_same_score(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    ucb = UCB(alpha=0.5)
    for i in range(node.n_children):
        if i != k-1:
            node.children[i].visit_count = 2
        else:
            node.children[i].visit_count = 1
    node.visit_count = 15
    assert (ucb(node) == (k-1))


@pytest.mark.parametrize("score, success", [(0, False), (1, True), (2, True), (4, True)])
def test_ucb_same_count(score, success) -> None:
    node = MCTSNode(label=0, rank=2, n_children=4)
    node.expand()
    ucb = UCB(alpha=0.5)
    k = 3
    for i in range(node.n_children):
        if i != k-1:
            node.children[i].visit_count = 1
            node.children[i].score = 0.5
        else:
            node.children[i].visit_count = 1
            node.children[i].score = score

    node.visit_count = 4
    assert ((ucb(node) == (k-1)) == success)


######################################################################
#                          Thompson Sampling                         #
######################################################################
def test_thompson_visited() -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    ucb = UCB(alpha=0.5)
    with pytest.raises(AssertionError):
        ucb(node)


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_thompson_no_visit(k) -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    thompson = Thompson_Sampling(a=1., b=1.)
    for i in range(node.n_children):
        if i != k-1:
            node.children[i].visit_count = 1
    assert (thompson(node) == (k-1))


@pytest.mark.parametrize("k, seed_out", [(1, 5), (2, 5), (4, 3), (8, 5)])
def test_thompson(k, seed_out) -> None:
    node = MCTSNode(label=0, rank=2, n_children=8)
    node.expand()
    thompson = Thompson_Sampling(a=1., b=1.)
    for i in range(node.n_children):
        if i != k-1:
            node.children[i].visit_count = 5
        else:
            node.children[i].visit_count = 1
    node.visit_count = 15
    np.random.seed(0)
    assert (thompson(node) == seed_out)
