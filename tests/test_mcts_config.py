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


def test_mcts_config_init(constraint) -> None:
    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    assert str(config.selection_policy) == str(EpsGreedy())
    assert str(config.exploration_policy) == str(Uniform())
    assert not config.normalize_scores
    assert config.normalization_option == NormOption.SOFTMAX
    assert config.n_classes == 3
    assert isinstance(config.constraint, Constraint)


def test_mcts_config_normalize_scores_no_option(constraint) -> None:
    with pytest.raises(ValueError):
        MCTSConfig(normalize_scores=True, n_classes=3, constraint=constraint)


def test_mcts_config_normalize_scores_with_option(constraint) -> None:
    config = MCTSConfig(normalize_scores=True, normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    assert config.normalize_scores
    assert config.normalization_option == NormOption.SOFTMAX


def test_mcts_config_save_load(constraint) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
        temp_file_name = temp.name

    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    config.save_config(temp_file_name)

    assert os.path.exists(temp_file_name)

    loaded_config = MCTSConfig(n_classes=1, constraint=constraint)
    loaded_config.load_config(temp_file_name)

    assert loaded_config.n_classes == config.n_classes
    assert loaded_config.constraint == config.constraint
    assert loaded_config.selection_policy == config.selection_policy
    assert loaded_config.exploration_policy == config.exploration_policy
    assert loaded_config.best_child_policy == config.best_child_policy
    assert loaded_config.normalize_scores == config.normalize_scores
    assert loaded_config.normalization_option == config.normalization_option

    os.remove(temp_file_name)


def test_monte_carlo_config_init(constraint) -> None:
    config = MonteCarloConfig(n_classes=3, constraint=constraint)
    assert str(config.selection_policy) == str(Uniform())
    assert str(config.exploration_policy) == str(Uniform())
    assert str(config.best_child_policy) == str(Greedy())
    assert not config.normalize_scores
    assert config.n_classes == 3
    assert isinstance(config.constraint, Constraint)


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

    assert loaded_config.n_classes == config.n_classes
    assert loaded_config.constraint == config.constraint
    assert loaded_config.selection_policy == config.selection_policy
    assert loaded_config.exploration_policy == config.exploration_policy
    assert loaded_config.best_child_policy == config.best_child_policy
    assert loaded_config.normalize_scores == config.normalize_scores

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
