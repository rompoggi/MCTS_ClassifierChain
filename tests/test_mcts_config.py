"""
Tests for the MCTSConfig class.
"""

import pytest

import os
import tempfile

from mcts_inference.mcts_config import MCTSConfig, MonteCarloConfig
from mcts_inference.utils import NormOption
from mcts_inference.constraints import Constraint
from mcts_inference.policy import EpsGreedy, Uniform, Greedy


@pytest.fixture
def constraint() -> Constraint:
    return Constraint(time=True, d_time=1.0)


def test_mcts_config_properties(constraint) -> None:
    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint, step_once=True)
    assert (config.n_classes == 3)
    assert (isinstance(config.constraint, Constraint))
    assert (config.constraint == constraint)

    assert (config.selection_policy == EpsGreedy())
    assert (config.exploration_policy == Uniform())
    assert (config.best_child_policy == Greedy())

    assert (config.normalize_scores is False)
    assert (config.normalization_option == NormOption.SOFTMAX)

    assert (config.parallel is True)
    assert (config.step_once is True)


def test_mcts_config_normalize_scores_no_option(constraint) -> None:
    with pytest.raises(ValueError):
        MCTSConfig(normalize_scores=True, n_classes=3, constraint=constraint)


def test_mcts_config_normalize_scores_with_option(constraint) -> None:
    config = MCTSConfig(normalize_scores=True, normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    assert (config.normalize_scores is True)
    assert (config.normalization_option == NormOption.SOFTMAX)


def test_mcts_config_str(constraint) -> None:
    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    expected_str: str = f"MCTSConfig(n_classes={config.n_classes}, constraint={config.constraint}, selection_policy={config.selection_policy}, " \
                        f"exploration_policy={config.exploration_policy}, best_child_policy={config.best_child_policy}, " \
                        f"normalize_scores={config.normalize_scores}, normalization_option={config.normalization_option}, " \
                        f"parallel={config.parallel}, step_once={config.step_once}, verbose={config.verbose}, " \
                        f"visualize_tree_graph={config.visualize_tree_graph}, save_tree_graph={config.save_tree_graph})"
    assert (str(config) == expected_str)


def test_mcts_config_repr(constraint) -> None:
    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    expected_repr: str = f"MCTSConfig(n_classes={config.n_classes}, constraint={config.constraint}, selection_policy={config.selection_policy}, " \
                         f"exploration_policy={config.exploration_policy}, best_child_policy={config.best_child_policy}, " \
                         f"normalize_scores={config.normalize_scores}, normalization_option={config.normalization_option}, " \
                         f"parallel={config.parallel}, step_once={config.step_once}, verbose={config.verbose}, " \
                         f"visualize_tree_graph={config.visualize_tree_graph}, save_tree_graph={config.save_tree_graph})"
    assert (repr(config) == expected_repr)


def test_mcts_config_eq(constraint) -> None:
    config1 = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    config2 = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    assert (config1 == config2)

    config3 = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    config4 = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=4, constraint=constraint)
    assert (config3 != config4)

    assert (config1 != "config1")


def test_mcts_config_save_load(constraint) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_file_name = temp.name

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    config.save_config(temp_file_name)

    assert (os.path.exists(temp_file_name))

    loaded_config = MCTSConfig(n_classes=1, constraint=constraint)
    loaded_config.load_config(temp_file_name)

    assert (loaded_config.n_classes == config.n_classes)
    assert (loaded_config.constraint == config.constraint)

    assert (loaded_config.selection_policy == config.selection_policy)
    assert (loaded_config.exploration_policy == config.exploration_policy)
    assert (loaded_config.best_child_policy == config.best_child_policy)

    assert (loaded_config.normalize_scores == config.normalize_scores)
    assert (loaded_config.normalization_option == config.normalization_option)

    assert (loaded_config.parallel == config.parallel)
    assert (loaded_config.step_once == config.step_once)

    assert (loaded_config.verbose == config.verbose)
    assert (loaded_config.visualize_tree_graph == config.visualize_tree_graph)
    assert (loaded_config.save_tree_graph == config.save_tree_graph)

    assert (loaded_config.loaded_from == temp_file_name)
    os.remove(temp_file_name)


def test_monte_carlo_config_init(constraint) -> None:
    config = MonteCarloConfig(n_classes=3, constraint=constraint)
    assert (config.n_classes == 3)
    assert (isinstance(config.constraint, Constraint))
    assert (config.constraint == constraint)

    assert (config.selection_policy == Uniform())
    assert (config.exploration_policy == Uniform())
    assert (config.best_child_policy == Greedy())

    assert (config.normalize_scores is False)

    assert (config.step_once is False)


def test_monte_carlo_config_save_load(constraint) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_file_name = temp.name

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    config = MonteCarloConfig(n_classes=3, constraint=constraint)
    config.save_config(temp_file_name)

    assert os.path.exists(temp_file_name)

    loaded_config = MonteCarloConfig(n_classes=1, constraint=constraint)
    loaded_config.load_config(temp_file_name)

    assert (loaded_config.n_classes == config.n_classes)
    assert (loaded_config.constraint == config.constraint)

    assert (loaded_config.selection_policy == config.selection_policy)
    assert (loaded_config.exploration_policy == config.exploration_policy)
    assert (loaded_config.best_child_policy == config.best_child_policy)

    assert (loaded_config.normalize_scores == config.normalize_scores)
    assert (loaded_config.normalization_option == config.normalization_option)

    assert (loaded_config.parallel == config.parallel)
    assert (loaded_config.step_once == config.step_once)

    assert (loaded_config.verbose == config.verbose)
    assert (loaded_config.visualize_tree_graph == config.visualize_tree_graph)
    assert (loaded_config.save_tree_graph == config.save_tree_graph)

    assert (loaded_config.loaded_from == temp_file_name)

    os.remove(temp_file_name)


def test_save_config_invalid_format(constraint) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_file_name = temp.name

    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)

    with pytest.raises(ValueError):
        config.save_config(temp_file_name, format='txt')

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    config.save_config(temp_file_name, format='json')

    config = MCTSConfig(path=temp_file_name)

    with pytest.raises(FileExistsError):
        config.save_config(path=temp_file_name, format='json')

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)


def test_load_config_invalid_format(constraint) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_file_name = temp.name

    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)

    with pytest.raises(ValueError):
        config.load_config(path=temp_file_name, format='txt')

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    with pytest.raises(FileNotFoundError):
        config.load_config(path=temp_file_name)

    with pytest.raises(FileNotFoundError):
        config = MCTSConfig(path=temp_file_name)
