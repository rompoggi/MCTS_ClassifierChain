"""
Test file for the ulitils module.
"""

import pytest
from mcts_inference.utils import debug, NormOption
from typing import Any


def test_normoption_values() -> None:
    assert NormOption.SOFTMAX.value == 1
    assert NormOption.UNIFORM.value == 2
    assert NormOption.NONE.value == 3


def test_normoption_names() -> None:
    assert NormOption(1) == NormOption.SOFTMAX
    assert NormOption(2) == NormOption.UNIFORM
    assert NormOption(3) == NormOption.NONE


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


@pytest.mark.skip(reason="example skip")
def test_skip() -> None:
    assert (True is False)


@pytest.mark.xfail(reason="example xfail")
def test_divide_by_zero() -> None:
    assert 1 / 0 == 1