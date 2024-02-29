from time import time
from typing import Any


class Constraint:
    """
    Constraint class to handle the time and iteration constraints.
    It is used for the MCTS algorithm to stop the search when the time or iteration constraints are reached.

    Args:
        time (bool): If True, the time constraint is activated
        max_iter (bool): If True, the iteration constraint is activated
        d_time (float): The time constraint in seconds
        n_iter (int): The iteration constraint
        verbose (bool): If True, the constraints will print a message when they are reached

    Methods:
        reset(self) -> None: Reset the constraints to their initial state
        _bool_time(self) -> bool: Check the time constraint
        _bool_iter(self) -> bool: Check the iteration constraint
        __bool__(self) -> bool: Returns True if the constraints are satisfied, False otherwise

    Raises:
        AssertionError: If both time and max_iter are False
        AssertionError: If max_iter is True and n_iter is not a positive integer
        AssertionError: If time is True and d_time is not a positive float

    Examples:
        >>> c = Constraint(time=True, d_time=1., max_iter=False, n_iter=0, verbose=True)
        >>> bool(c)
        True
        >>> c = Constraint(time=False, d_time=1., max_iter=True, n_iter=100, verbose=True)
        >>> bool(c)
        True
        >>> c = Constraint(time=True, d_time=1., max_iter=True, n_iter=100, verbose=True)
        >>> bool(c)
        True
        >>> c.curr_iter == 1
        True
        >>> c = Constraint(time=False, d_time=1., max_iter=False, n_iter=0, verbose=True)
        AssertionError: At least time=False or max_iter=False should be True
    """
    def __init__(self, time: bool = False, max_iter: bool = False, d_time: float = 1., n_iter: int = 100, verbose: bool = False) -> None:
        assert (time or max_iter), f"At least {time=} or {max_iter=} should be True"
        assert ((not max_iter) or (isinstance(n_iter, int) and n_iter > 0)), f"{n_iter=} should be positive if {max_iter=}"
        assert ((not time) or (d_time > 0)), f"{d_time=} should be positive if {time=}"

        self.time: bool = time
        self.d_time: float = d_time
        self.end_time: float = 0

        self.max_iter: bool = max_iter
        self.n_iter: int = n_iter
        self.curr_iter: int = 0

        self.reset()

        self.verbose: bool = verbose

    def reset(self) -> None:
        """
        Reset the constraints to their initial state
        """
        self.end_time = time() + self.d_time
        self.curr_iter = 0

    def _bool_time(self) -> bool:
        """
        Check the time constraint
        """
        return (not self.time or self.end_time >= time())

    def _bool_iter(self) -> bool:
        """
        Check the iteration constraint
        """
        self.curr_iter += 1
        return (not self.max_iter or self.curr_iter <= self.n_iter)

    def __bool__(self) -> bool:
        """
        Returns True if the constraints are satisfied, False otherwise
        It updates the current iteration and time based on the options
        """
        if self.verbose:  # verbose
            bt: bool = self._bool_time()
            bi: bool = self._bool_iter()
            if not bt:
                print(f"Time Constraint Attained. Current iteration: {self.curr_iter:_}/{self.n_iter:_}")
                return False
            if not bi:
                print(f"Iteration Constraint Attained. Time left: {self.end_time - time():.3f}/{self.d_time}s")
                return False
            return True
        return self._bool_time() and self._bool_iter()

    def __str__(self) -> str:
        """
        String representation of the constraint
        """
        if self.time:
            if self.max_iter:
                return f"Time={self.time}, MaxIter={self.max_iter}, d_time={self.d_time}, n_iter={self.n_iter}, \
curr_iter={self.curr_iter}, verbose={self.verbose}"
            return f"Time={self.time}, d_time={self.d_time}s, verbose={self.verbose}"
        return f"MaxIter={self.max_iter}, n_iter={self.n_iter}, curr_iter={self.curr_iter}, verbose={self.verbose}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Constraint):
            return False
        return self.to_dict() == __value.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            'time': self.time,
            'd_time': self.d_time,
            'verbose': self.verbose,
            'max_iter': self.max_iter,
            'n_iter': self.n_iter,
        }


__all__: list[str] = ['Constraint']
