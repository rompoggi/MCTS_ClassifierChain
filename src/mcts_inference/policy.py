from .mcts import MCTSNode, randmax
import numpy as np
from typing import Any


class Policy:
    def __init__(self) -> None:
        pass

    def __call__(self, node: MCTSNode) -> Any:
        raise NotImplementedError("Policy.__call__ method not implemented.")


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
        if np.random.rand() < self.epsilon:  # explore
            return np.random.choice(node.n_children)
        return randmax(node.get_children_scores())

    def name(self) -> str:
        return "EpsGreedy"

    def __str__(self) -> str:
        return f"EpsGreedy(epsilon={self.epsilon})"


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

        if min([child.visit_count for child in node.children]) == 0:
            return randmax([-child.visit_count for child in node.children])

        ucb: np.ndarray[Any, np.dtype[Any]] = np.array([child.score +
                                                        np.sqrt(self.alpha * np.log(node.visit_count) / child.visit_count)
                                                        for child in node.children])
        return randmax(ucb)

    def name(self) -> str:
        return "UCB"

    def __str__(self) -> str:
        return f"UCB(alpha={self.alpha})"


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
        if min(node.get_children_counts()) == 0:
            return randmax(-node.get_children_counts())

        rwd: np.ndarray[Any, np.dtype[np.float64]] = node.get_children_scores()
        pi: np.ndarray[Any, np.dtype[np.float64]] = np.random.beta(rwd + self.a, self.b + node.get_children_counts() - rwd)
        return randmax(pi)

    def name(self) -> str:
        return "Thompson Sampling"

    def __str__(self) -> str:
        return f"Thompson_Sampling(a={self.a}, b={self.b})"


__all__: list[str] = ["Policy", "Uniform", "EpsGreedy", "UCB", "Thompson_Sampling"]
