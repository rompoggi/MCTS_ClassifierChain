# TODO: Write tests

from unittest import mock
from mcts_inference.brute_force import BF_wrapper
from mcts_inference.mcts import MCTSConfig
from mcts_inference.constraints import Constraint


def test_BF_wrapper() -> None:
    config = MCTSConfig(n_classes=3, constraint=Constraint(time=True, d_time=0.2))
    # Mock the MCTS_one_step_atime function
    with mock.patch('mcts_inference.brute_force.BF', return_value=[1, 2, 3]) as mock_BF_wrapper:
        result = BF_wrapper((1, 'a', config))
        mock_BF_wrapper.assert_called_once_with(1, 'a', config)
        assert result == [1, 2, 3], "BF_wrapper should return the same result as BF"

    with mock.patch('mcts_inference.brute_force.BF', return_value=[4, 5, 6]) as mock_BF_wrapper:
        result = BF_wrapper(('test', 2.5, config))
        mock_BF_wrapper.assert_called_once_with('test', 2.5, config)
        assert result == [4, 5, 6], "BF_wrapper should return the same result as BF"
