"""
Test file for the pcc module.
"""

from unittest import mock

from mcts_inference.pcc import PCC_wrapper
from mcts_inference.mcts_config import MCTSConfig
from mcts_inference.constraints import Constraint


def test_PCC_wrapper() -> None:
    config = MCTSConfig(n_classes=3, constraint=Constraint(time=True, d_time=0.2))
    with mock.patch('mcts_inference.pcc._PCC', return_value=[1, 2, 3]) as mock_PCC_wrapper:
        result = PCC_wrapper((1, 'a', config))
        mock_PCC_wrapper.assert_called_once_with(1, 'a', config)
        assert result == [1, 2, 3], "PCC_wrapper should return the same result as _PCC"

    with mock.patch('mcts_inference.pcc._PCC', return_value=[4, 5, 6]) as mock_PCC_wrapper:
        result = PCC_wrapper(('test', 2.5, config))
        mock_PCC_wrapper.assert_called_once_with('test', 2.5, config)
        assert result == [4, 5, 6], "PCC_wrapper should return the same result as _PCC"
