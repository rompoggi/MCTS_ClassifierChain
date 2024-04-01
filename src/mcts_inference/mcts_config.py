"""
This module defines the MCTSConfig class which represents the configuration for the Monte Carlo Tree Search (MCTS) algorithm.

The MCTSConfig class allows users to specify various parameters and policies for the MCTS algorithm, such as the number of classes,
time constraint, selection policy, exploration policy, best child policy, normalization options, parallel execution, verbosity, and more.

Example usage:
    config = MCTSConfig(n_classes=2, constraint=Constraint(time=True, d_time=1.0))
    config.selection_policy = EpsGreedy(epsilon=0.2)
    config.exploration_policy = Uniform()
    config.best_child_policy = Greedy()
    config.normalize_scores = False
    config.parallel = True
    config.verbose = False

    # Save the configuration to a file
    config.save_config('/path/to/config.json')

    # Load the configuration from a file
    config.load_config('/path/to/config.json')

    # Print the configuration
    print(config)
"""

from typing import Dict, Optional, Any
import os
import json

from .utils import NormOption
from .constraints import Constraint
from .policy import Policy, EpsGreedy, Uniform, Greedy


class MCTSConfig:
    def __init__(self,
                 n_classes: int = 2,
                 constraint: Constraint = Constraint(time=True, d_time=1.),

                 selection_policy: Policy = EpsGreedy(epsilon=0.2),
                 exploration_policy: Policy = Uniform(),
                 best_child_policy: Policy = Greedy(),

                 normalize_scores: bool = False,
                 normalization_option: Optional[NormOption] = None,

                 step_once: bool = True,
                 parallel: bool = True,

                 verbose: bool = False,
                 visualize_tree_graph: bool = False,
                 save_tree_graph: bool = False,

                 path: Optional[str] = None,
                 format='json',
                 ) -> None:
        """
        Initializes a new instance of the MCTSConfig class.

        Args:
            n_classes (int, optional): The number of classes. Defaults to 2.
            constraint (Constraint, optional): The time constraint for the MCTS algorithm. Defaults to Constraint(time=True, d_time=1.0).
            selection_policy (Policy, optional): The selection policy for MCTS. Defaults to EpsGreedy(epsilon=0.2).
            exploration_policy (Policy, optional): The exploration policy for MCTS. Defaults to Uniform().
            best_child_policy (Policy, optional): The best child policy for MCTS. Defaults to Greedy().
            normalize_scores (bool, optional): Whether to normalize scores. Defaults to False.
            normalization_option (NormOption, optional): The normalization option for scores. Defaults to None.
            step_once (bool, optional): Whether to perform MCTS in a step-by-step manner. Defaults to True.
            parallel (bool, optional): Whether to perform MCTS in parallel. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            visualize_tree_graph (bool, optional): Whether to visualize the MCTS tree graph. Defaults to False.
            save_tree_graph (bool, optional): Whether to save the MCTS tree graph. Defaults to False.
            path (str, optional): The path to load/save the configuration. Defaults to None.
            format (str, optional): The format to load/save the configuration. Only 'json' is supported. Defaults to 'json'.
        """
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

        self._parallel: bool = parallel
        self._step_once: bool = step_once

        self._verbose: bool = verbose
        self._visualize_tree_graph: bool = visualize_tree_graph
        self._save_tree_graph: bool = save_tree_graph

        self._loaded_from: Optional[str] = None

#######
    @property
    def n_classes(self) -> int:
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value) -> None:
        self._n_classes = value

#######
    @property
    def constraint(self) -> Constraint:
        return self._constraint

    @constraint.setter
    def constraint(self, value) -> None:
        self._constraint = Constraint(**value)

    def replace_constraint(self, constraint: Constraint) -> None:
        self._constraint = constraint

#######
    @property
    def selection_policy(self) -> Policy:
        return self._selection_policy

    @selection_policy.setter
    def selection_policy(self, value) -> None:
        self._selection_policy = eval(value['repr'])

#######
    @property
    def exploration_policy(self) -> Policy:
        return self._exploration_policy

    @exploration_policy.setter
    def exploration_policy(self, value) -> None:
        self._exploration_policy = eval(value['repr'])

#######
    @property
    def best_child_policy(self) -> Policy:
        return self._best_child_policy

    @best_child_policy.setter
    def best_child_policy(self, value) -> None:
        self._best_child_policy = eval(value['repr'])

#######
    @property
    def normalize_scores(self) -> bool:
        return self._normalize_scores

    @normalize_scores.setter
    def normalize_scores(self, value) -> None:
        self._normalize_scores = value

#######
    @property
    def normalization_option(self) -> NormOption | None:
        return self._normalization_option

    @normalization_option.setter
    def normalization_option(self, value) -> None:
        self._normalization_option = NormOption(value) if (value is not None) else None

#######
    @property
    def parallel(self) -> bool:
        return self._parallel

    @parallel.setter
    def parallel(self, value: bool) -> None:
        self._parallel = value

#######
    @property
    def step_once(self) -> bool:
        return self._step_once

    @step_once.setter
    def step_once(self, value: bool) -> None:
        self._step_once = value

#######
    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

#######
    @property
    def loaded_from(self) -> Optional[str]:
        return self._loaded_from

    @loaded_from.setter
    def loaded_from(self, value: Optional[str]) -> None:
        self._loaded_from = value

#######
    @property
    def visualize_tree_graph(self) -> bool:
        return self._visualize_tree_graph

    @visualize_tree_graph.setter
    def visualize_tree_graph(self, value: bool) -> None:
        self._visualize_tree_graph = value

#######
    @property
    def save_tree_graph(self) -> bool:
        return self._save_tree_graph

    @save_tree_graph.setter
    def save_tree_graph(self, value: bool) -> None:
        self._save_tree_graph = value

#################################################################################
    def __str__(self) -> str:
        return f"MCTSConfig(n_classes={self.n_classes}, constraint={self.constraint}, selection_policy={self.selection_policy}, " \
               f"exploration_policy={self.exploration_policy}, best_child_policy={self.best_child_policy}, " \
               f"normalize_scores={self.normalize_scores}, normalization_option={self.normalization_option}, " \
               f"parallel={self.parallel}, step_once={self.step_once}, verbose={self.verbose}, visualize_tree_graph={self.visualize_tree_graph}, " \
               f"save_tree_graph={self.save_tree_graph})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (self.n_classes == other.n_classes) and \
               (self.constraint == other.constraint) and \
               (self.selection_policy == other.selection_policy) and \
               (self.exploration_policy == other.exploration_policy) and \
               (self.best_child_policy == other.best_child_policy) and \
               (self.normalize_scores == other.normalize_scores) and \
               (self.normalization_option == other.normalization_option) and \
               (self.parallel == other.parallel) and \
               (self.step_once == other.step_once) and \
               (self.verbose == other.verbose) and \
               (self.visualize_tree_graph == other.visualize_tree_graph) and \
               (self.save_tree_graph == other.save_tree_graph)

#################################################################################
    def save_config(self, path, format='json') -> None:
        if format != 'json':
            raise ValueError("Unsupported format. Only 'json' is supported.")

        if os.path.exists(path):
            raise FileExistsError("File already exists. Please provide a different path or delete the existing file.")

        config_dict: Dict[str, Any] = {
            'n_classes': self.n_classes,
            'constraint': self.constraint.to_dict(),

            'selection_policy': self.selection_policy.to_dict(),
            'exploration_policy': self.exploration_policy.to_dict(),
            'best_child_policy': self.best_child_policy.to_dict(),

            'normalize_scores': self.normalize_scores,
            'normalization_option': self.normalization_option.value if self.normalization_option is not None else None,

            'parallel': self.parallel,
            'step_once': self.step_once,

            'verbose': self.verbose,
            'visualize_tree_graph': self.visualize_tree_graph,
            'save_tree_graph': self.save_tree_graph,

            'loaded_from': self.loaded_from,
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

        setattr(self, "loaded_from", path)


class MonteCarloConfig(MCTSConfig):
    def __init__(self, n_classes: int, constraint: Constraint) -> None:
        super().__init__(n_classes, constraint)
        self._selection_policy = Uniform()
        self._exploration_policy = Uniform()
        self._best_child_policy = Greedy()
        self._normalize_scores = False
        self._step_once = False
