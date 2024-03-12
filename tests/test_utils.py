"""
Test file for the utils module.
"""

import pytest
from mcts_inference.utils import randmax, NormOption, debug, deprecated
from typing import Any


# Randmax function tests
@pytest.mark.parametrize("arr, argmax", [([0, 0, 1], 2), ([0, 10, -1], 1)])
def test_randmax_unique_max(arr, argmax) -> None:
    assert (randmax(arr) == argmax)


@pytest.mark.parametrize("arr, argmaxs", [([0, 1, 1, 0, 0], [1, 2]), ([11, 10, -1, 11, 0], [0, 3])])
def test_randmax_mult_max(arr, argmaxs) -> None:
    assert (randmax(arr) in argmaxs)


# NormOption tests
def test_normoption_values() -> None:
    assert NormOption.SOFTMAX.value == 1
    assert NormOption.UNIFORM.value == 2
    assert NormOption.NONE.value == 3


def test_normoption_names() -> None:
    assert NormOption(1) == NormOption.SOFTMAX
    assert NormOption(2) == NormOption.UNIFORM
    assert NormOption(3) == NormOption.NONE


# Debug tests
def test_debug(capfd) -> None:
    @debug
    def f(x: int, y: int, z: int) -> int:
        return x * y + z

    assert f(1, 2, 3) == 5
    captured: Any = capfd.readouterr()
    assert captured.out == "'f': args=(1, 2, 3), kwargs={}\n'f': output=5\n"

    assert f(25, 2, -5) == 45
    captured = capfd.readouterr()
    assert captured.out == "'f': args=(25, 2, -5), kwargs={}\n'f': output=45\n"

    # -----
    @debug
    def g(x: int, y: int) -> int:
        return x * y

    assert g(2, 3) == 6
    captured = capfd.readouterr()
    assert captured.out == "'g': args=(2, 3), kwargs={}\n'g': output=6\n"

    assert g(0, 1) == 0
    captured = capfd.readouterr()
    assert captured.out == "'g': args=(0, 1), kwargs={}\n'g': output=0\n"

# Deprecated function tests
def test_deprecated(caplog) -> None:
    @deprecated
    def deprecated_func(x: int, y: int) -> int:
        return x + y

    assert deprecated_func(1, 2) == 3
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'WARNING'
    assert "Call to deprecated function deprecated_func." in caplog.records[0].message

    assert deprecated_func(3, 4) == 7
    assert len(caplog.records) == 2
    assert caplog.records[1].levelname == 'WARNING'
    assert "Call to deprecated function deprecated_func." in caplog.records[1].message
