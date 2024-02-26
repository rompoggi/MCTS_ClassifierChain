from .mcts import MCTSNode, randmax
import numpy as np
from typing import Any


class Policy:
    def __init__(self) -> None:
        pass

    def __call__(self: Any, *args, **kwargs) -> Any:
        raise NotImplementedError("Policy.__call__ method not implemented.")


class Random(Policy):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, node: MCTSNode) -> Any:
        """
        Random policy to select the next node to visit.

        Args:
            node (MCTSNode): The node from which to select the next node

        Examples:
            >>> node = MCTSNode(label=0, rank=2, n_children=2, score=0.5, parent=None, parent_labels=[])
            >>> node.expand()
            >>> random(node)
            0
        """
        return np.random.choice(node.n_children)


class EpsGreedy(Policy):
    def __init__(self, epsilon: float = 0.2) -> None:
        super().__init__()
        assert (epsilon >= 0 and epsilon <= 1), f"{epsilon = } should be in the [0,1] range."
        self.epsilon: float = epsilon

    def __call__(self, node: MCTSNode) -> Any:
        if np.random.rand() < self.epsilon:  # explore
            return np.random.choice(node.n_children)
        return randmax(node.get_children_scores())


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

