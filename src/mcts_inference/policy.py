"""
Policy module
Implement the different policies to select the next node to visit in the MCTS algorithm.
Policies:
- Uniform
- Greedy
- EpsGreedy
- UCB
- Thompson_Sampling

Examples:
- Create a Uniform policy:
>>> uniform = Uniform()
>>> print(uniform)
Uniform

- Create a Greedy policy:
>>> greedy = Greedy()
>>> print(greedy)
Greedy

- Create an EpsGreedy policy:
>>> eps_greedy = EpsGreedy(epsilon=0.1)
>>> print(eps_greedy)
EpsGreedy(epsilon=0.1)

- Create a UCB policy:
>>> ucb = UCB(alpha=0.5)
>>> print(ucb)
UCB(alpha=0.5)

- Create a Thompson_Sampling policy:
>>> ts = Thompson_Sampling(a=1., b=1.)
>>> print(ts)
Thompson_Sampling(a=1., b=1.)

- Select the next node to visit using any of the policy:
>>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5)
>>> node.expand()
>>> node[1].score = 0.5
>>> eps_greedy(node)
1
>>> ucb(node)
1
"""

import numpy as np
from typing import Any

from .mcts_node import MCTSNode
from .utils import randmax


class Policy:
    def __init__(self) -> None:
        pass

    def __call__(self, node: MCTSNode) -> Any:
        raise NotImplementedError("Policy.__call__ method not implemented.")

    def name(self) -> str:
        raise NotImplementedError("Policy.name method not implemented.")

    def __str__(self) -> str:
        raise NotImplementedError("Policy.__str__ method not implemented.")

    def __repr__(self) -> str:
        raise NotImplementedError("Policy.__repr__ method not implemented.")


class Uniform(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, node: MCTSNode) -> Any:
        """
        Uniform policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> uniform = Uniform()
            >>> uniform(node)
            0
             >>> uniform(node)
            1
        """
        return np.random.choice(node.n_children)

    def name(self) -> str:
        return "Uniform"

    def __str__(self) -> str:
        return "Uniform"

    def __repr__(self) -> str:
        return str(self)


class Greedy(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, node: MCTSNode) -> Any:
        """
        Greedy policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node.children[1].score = 0.5
            >>> greedy = Greedy()
            >>> greedy(node)
            1
        """
        return randmax(node.get_children_scores())

    def name(self) -> str:
        return "Greedy"

    def __str__(self) -> str:
        return "Greedy"

    def __repr__(self) -> str:
        return str(self)


class EpsGreedy(Policy):
    """
    Epislon-Greedy policy to select the next node to visit.
    """
    def __init__(self, epsilon: float = 0.2) -> None:
        super().__init__()
        assert (epsilon >= 0 and epsilon <= 1), f"{epsilon = } should be in the [0,1] range."
        self.epsilon: float = epsilon

    def __call__(self, node: MCTSNode) -> Any:
        """
        Epislon-Greedy policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
        >>> node.expand()
        >>> node.children[1].score = 0.5
        >>> eg = EpsGreedy(epsilon=0.5)
        >>> eg(node)
        1
        """
        if np.random.rand() <= self.epsilon:  # explore
            return np.random.choice(node.n_children)
        return randmax(node.get_children_scores())

    def name(self) -> str:
        return "EpsGreedy"

    def __str__(self) -> str:
        return f"EpsGreedy(epsilon={self.epsilon})"

    def __repr__(self) -> str:
        return str(self)


class UCB(Policy):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha: float = alpha

    def __call__(self, node: MCTSNode) -> Any:
        """
        Upper Confidence Bound (UCB) policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node.children[1].score = 0.5
            >>> ucb = UCB(alpha=0.5)
            >>> ucb(node)
            1
        """
        assert (node.visit_count > 0), "Node has not yet been visited. A problem appened."

        counts: np.ndarray[Any, np.dtype[np.int64]] = node.get_children_counts()
        if min(counts) == 0:
            return randmax(-counts)

        scores: np.ndarray[Any, np.dtype[np.float64]] = node.get_children_scores()
        ucb: np.ndarray[Any, np.dtype[np.float64]] = scores / counts + np.sqrt(self.alpha * np.log(node.visit_count) / counts)
        return randmax(ucb)

    def name(self) -> str:
        return "UCB"

    def __str__(self) -> str:
        return f"UCB(alpha={self.alpha})"

    def __repr__(self) -> str:
        return str(self)


class Thompson_Sampling(Policy):
    def __init__(self, a: float = 1., b: float = 1.) -> None:
        super().__init__()
        self.a: float = a
        self.b: float = b

    def __call__(self, node: MCTSNode) -> Any:
        """
        Thompson Sampling policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> node.children[1].score = 0.5
            >>> ts = Thompson_Sampling(a=1., b=1.)
            >>> ts(node)
            1
        """

        counts: np.ndarray[Any, np.dtype[np.int64]] = node.get_children_counts()
        if min(counts) == 0:
            return randmax(-counts)

        scores: np.ndarray[Any, np.dtype[np.float64]] = node.get_children_scores()
        pi: np.ndarray[Any, np.dtype[np.float64]] = np.random.beta(scores + self.a, self.b + counts - scores)
        return randmax(pi)

    def name(self) -> str:
        return "Thompson_Sampling"

    def __str__(self) -> str:
        return f"Thompson_Sampling(a={self.a}, b={self.b})"

    def __repr__(self) -> str:
        return str(self)


__all__: list[str] = ["Policy", "Uniform", "Greedy", "EpsGreedy", "UCB", "Thompson_Sampling"]
