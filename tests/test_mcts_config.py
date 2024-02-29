"""
Tests for the MCTSConfig class.
"""

import pytest
import os

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
    if os.path.exists('/tmp/config.json'):
        os.remove('/tmp/config.json')

    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)
    config.save_config('/tmp/config.json')

    assert os.path.exists('/tmp/config.json')

    loaded_config = MCTSConfig(n_classes=1, constraint=constraint)
    loaded_config.load_config('/tmp/config.json')

    assert loaded_config.n_classes == config.n_classes
    assert loaded_config.constraint == config.constraint
    assert loaded_config.selection_policy == config.selection_policy
    assert loaded_config.exploration_policy == config.exploration_policy
    assert loaded_config.best_child_policy == config.best_child_policy
    assert loaded_config.normalize_scores == config.normalize_scores
    assert loaded_config.normalization_option == config.normalization_option

    os.remove('/tmp/config.json')


def test_monte_carlo_config_init(constraint) -> None:
    config = MonteCarloConfig(n_classes=3, constraint=constraint)
    assert str(config.selection_policy) == str(Uniform())
    assert str(config.exploration_policy) == str(Uniform())
    assert str(config.best_child_policy) == str(Greedy())
    assert not config.normalize_scores
    assert config.n_classes == 3
    assert isinstance(config.constraint, Constraint)


def test_monte_carlo_config_save_load(constraint) -> None:
    if os.path.exists('/tmp/monte_carlo_config.json'):
        os.remove('/tmp/monte_carlo_config.json')

    config = MonteCarloConfig(n_classes=3, constraint=constraint)
    config.save_config('/tmp/monte_carlo_config.json')

    assert os.path.exists('/tmp/monte_carlo_config.json')

    loaded_config = MonteCarloConfig(n_classes=1, constraint=constraint)
    loaded_config.load_config('/tmp/monte_carlo_config.json')

    assert loaded_config.n_classes == config.n_classes
    assert loaded_config.constraint == config.constraint
    assert loaded_config.selection_policy == config.selection_policy
    assert loaded_config.exploration_policy == config.exploration_policy
    assert loaded_config.best_child_policy == config.best_child_policy
    assert loaded_config.normalize_scores == config.normalize_scores

    os.remove('/tmp/monte_carlo_config.json')


def test_save_config_invalid_format(constraint) -> None:
    config = MCTSConfig(normalization_option=NormOption.SOFTMAX, n_classes=3, constraint=constraint)

    with pytest.raises(ValueError):
        config.save_config('/tmp/config.json', format='txt')

    if os.path.exists('/tmp/config.txt'):
        os.remove('/tmp/config.txt')

    config.save_config('/tmp/config.json', format='json')

    with pytest.raises(FileExistsError):
        config.save_config(path='/tmp/config.json', format='json')

    if os.path.exists('/tmp/config.txt'):
        os.remove('/tmp/config.txt')


def test_load_config_invalid_format(constraint) -> None:
    config = MCTSConfig(n_classes=1, constraint=constraint, path='/tmp/config.json', format='json')

    with pytest.raises(ValueError):
        config.load_config(path='/tmp/config.json', format='txt')

    if os.path.exists('/tmp/config.json'):
        os.remove('/tmp/config.json')

    with pytest.raises(FileNotFoundError):
        config.load_config(path='/tmp/config.json')

    if os.path.exists('/tmp/config.txt'):
        os.remove('/tmp/config.txt')
