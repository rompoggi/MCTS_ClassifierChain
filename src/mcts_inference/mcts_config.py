"""
File to store the configuration of the MCTS algorithm.
"""

from typing import Optional
import os
import json

from .utils import NormOption
from .constraints import Constraint
from .policy import Policy, EpsGreedy, Uniform, Greedy


class MCTSConfig:
    def __init__(self,
                 n_classes: int = 2,
                 constraint: Constraint = Constraint(time=True, d_time=1.),
                 selection_policy: Policy = EpsGreedy(),
                 exploration_policy: Policy = Uniform(),
                 best_child_policy: Policy = Greedy(),
                 normalize_scores: bool = False,
                 normalization_option: Optional[NormOption] = None,
                 path=None,
                 format='json',
                 ) -> None:

        if path is not None:
            self.load_config(path, format)
            return

        assert (n_classes > 0), "Number of classes must be greater than 0"
        self._n_classes: int = n_classes
        self._constraint: Constraint = constraint

        self._selection_policy: Policy = selection_policy
        self._exploration_policy: Policy = exploration_policy
        self._best_child_policy: Policy = best_child_policy

        self._normalize_scores: bool = normalize_scores
        if self.normalize_scores and normalization_option is None:
            raise ValueError("Normalization option must be provided if normalize_scores is set to True")

        self._normalization_option: Optional[NormOption] = NormOption(normalization_option) if (normalization_option is not None) else None

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value):
        self._n_classes = value

    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @constraint.setter
    def constraint(self, value):
        self._constraint = Constraint(**value)

    @property
    def selection_policy(self) -> Policy:
        return self._selection_policy

    @selection_policy.setter
    def selection_policy(self, value):
        self._selection_policy = eval(value['repr'])

    @property
    def exploration_policy(self) -> Policy:
        return self._exploration_policy

    @exploration_policy.setter
    def exploration_policy(self, value):
        self._exploration_policy = eval(value['repr'])

    @property
    def best_child_policy(self) -> Policy:
        return self._best_child_policy

    @best_child_policy.setter
    def best_child_policy(self, value):
        self._best_child_policy = eval(value['repr'])

    @property
    def normalize_scores(self) -> bool:
        return self._normalize_scores

    @normalize_scores.setter
    def normalize_scores(self, value):
        self._normalize_scores = value

    @property
    def normalization_option(self) -> NormOption | None:
        return self._normalization_option

    @normalization_option.setter
    def normalization_option(self, value):
        self._normalization_option = NormOption(value) if (value is not None) else None

    def save_config(self, path, format='json') -> None:
        if format != 'json':
            raise ValueError("Unsupported format. Only 'json' is supported.")

        if os.path.exists(path):
            raise FileExistsError("File already exists. Please provide a different path or delete the existing file.")

        config_dict = {
            'n_classes': self.n_classes,
            'constraint': self.constraint.to_dict(),
            'selection_policy': self.selection_policy.to_dict(),
            'exploration_policy': self.exploration_policy.to_dict(),
            'best_child_policy': self.best_child_policy.to_dict(),
            'normalize_scores': self.normalize_scores,
            'normalization_option': self.normalization_option.value if self.normalization_option is not None else None
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f)

    def load_config(self, path, format='json') -> None:
        if format != 'json':
            raise ValueError("Unsupported format. Only 'json' is supported.")

        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist. Please provide a valid path.")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        for key, value in config_dict.items():
            setattr(self, key, value)


class MonteCarloConfig(MCTSConfig):
    def __init__(self, n_classes: int, constraint: Constraint) -> None:
        super().__init__(n_classes, constraint)
        self._selection_policy = Uniform()
        self._exploration_policy = Uniform()
        self._best_child_policy = Greedy()
        self._normalize_scores = False
