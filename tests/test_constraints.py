from mcts_inference.constraints import Constraint
from math import inf
import pytest


def test_constraint_initialization_off() -> None:
    """
    Try time=False and max_iter=False
    """
    try:
        Constraint(time=False, max_iter=False)
        assert (False), "At least one constraint should be on, but AssertionError was not raised"
    except Exception:
        assert (True), "AssertionError was raised correctly"


@pytest.mark.parametrize("test_input", [0, -1, -100])
def test_multi_n_iter(test_input) -> None:
    try:
        Constraint(max_iter=True, n_iter=test_input)
        assert (False), "Should not have non-positive n_iter if max_iter=True"
    except Exception:
        assert (True), "AssertionError was raised correctly"


@pytest.mark.parametrize("test_input", ["0", inf, -100, 0.1])
def test_multi_type(test_input) -> None:
    try:
        Constraint(max_iter=True, n_iter=test_input)
        assert (False), "Should not have non-positive n_iter if max_iter=True"
    except Exception:
        assert (True), "AssertionError was raised correctly"


@pytest.mark.parametrize("test_input", [0., -1, -100, -inf])
def test_constraint_initialization_dtime(test_input) -> None:
    try:
        Constraint(time=True, d_time=test_input)
        assert (False), "Time constraintmust be positive, but AssertionError was not raised"
    except Exception:
        assert (True), "AssertionError was raised correctly"


@pytest.mark.parametrize("test_input", [1., 0.1])
def test_constraint_time_attained(test_input) -> None:
    from time import sleep
    constraint = Constraint(time=True, d_time=test_input)
    if not bool(constraint):
        assert (False), "Constraint should be True, but it is False"
    sleep(test_input)
    assert (bool(constraint) is False), "Constraint should have been attained"


@pytest.mark.parametrize("test_input", [1, 3, 7, 1000000])
def test_constraint_iter_attained(test_input) -> None:
    constraint = Constraint(max_iter=True, n_iter=test_input)
    for _ in range(test_input):
        if not bool(constraint):
            assert (False), "Constraint should be True, but it is False"
    assert (bool(constraint) is False)


def test_bool_verbose(capfd) -> None:
    from time import sleep
    c = Constraint(time=False, max_iter=True, d_time=2., n_iter=1, verbose=True)
    assert bool(c), "Constraint should be True"
    assert not c.__bool__(), "Constraint should be False"
    captured = capfd.readouterr()
    assert captured.out[:41] == "Iteration Constraint Attained. Time left:"

    c = Constraint(time=True, max_iter=True, d_time=0.1, n_iter=3, verbose=True)
    sleep(0.1)
    bool(c)
    captured = capfd.readouterr()
    assert captured.out == f"Time Constraint Attained. Current iteration: {1}/{3}\n"


def test_reset() -> None:
    from time import sleep
    from time import time
    c = Constraint(time=True, max_iter=True, d_time=1., n_iter=3, verbose=True)
    sleep(0.5)  # Sleep for a second to ensure that time() increases
    c.reset()
    assert abs(c.end_time - (time() + c.d_time)) < 0.01  # Allow for a small error due to the time it takes to execute the code
    assert c.curr_iter == 0


@pytest.mark.parametrize("verbose_input", [True, False])
def test_constraint_str(verbose_input) -> None:
    c = Constraint(time=True, max_iter=True, d_time=2., n_iter=3, verbose=verbose_input)
    expected_str: str = f"Time={True}, MaxIter={True}, d_time={2.}, n_iter={3}, curr_iter={0}, verbose={verbose_input}"
    assert str(c) == expected_str

    bool(c)
    expected_str = f"Time={True}, MaxIter={True}, d_time={2.}, n_iter={3}, curr_iter={1}, verbose={verbose_input}"
    assert str(c) == expected_str

    # -----
    c = Constraint(time=False, max_iter=True, n_iter=3, verbose=verbose_input)
    expected_str = f"MaxIter={True}, n_iter={3}, curr_iter={0}, verbose={verbose_input}"
    assert str(c) == expected_str

    # -----
    c = Constraint(time=True, max_iter=False, d_time=1., verbose=verbose_input)
    expected_str = f"Time={True}, d_time={1.}s, verbose={verbose_input}"
    assert str(c) == expected_str


@pytest.mark.parametrize("verbose_input", [True, False])
def test_constraint_repr(verbose_input) -> None:
    c = Constraint(time=True, max_iter=True, d_time=2., n_iter=3, verbose=verbose_input)
    expected_str: str = f"Time={True}, MaxIter={True}, d_time={2.}, n_iter={3}, curr_iter={0}, verbose={verbose_input}"
    assert repr(c) == expected_str

    bool(c)
    expected_str = f"Time={True}, MaxIter={True}, d_time={2.}, n_iter={3}, curr_iter={1}, verbose={verbose_input}"
    assert repr(c) == expected_str

    # -----
    c = Constraint(time=False, max_iter=True, n_iter=3, verbose=verbose_input)
    expected_str = f"MaxIter={True}, n_iter={3}, curr_iter={0}, verbose={verbose_input}"
    assert repr(c) == expected_str

    bool(c)
    bool(c)

    expected_str = f"MaxIter={True}, n_iter={3}, curr_iter={2}, verbose={verbose_input}"
    assert repr(c) == expected_str

    # -----
    c = Constraint(time=True, max_iter=False, d_time=1., verbose=verbose_input)
    expected_str = f"Time={True}, d_time={1.}s, verbose={verbose_input}"
    assert repr(c) == expected_str


@pytest.mark.skip(reason="example skip")
def test_skip() -> None:
    assert (True is False)


@pytest.mark.xfail(reason="example xfail")
def test_divide_by_zero() -> None:
    assert 1 / 0 == 1
