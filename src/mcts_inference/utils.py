"""
Utility functions and classes for the MCTS algorithm.

Debug is a wrapper to help in debugging any function

NormOption is an enum to represent the different normalization options for the MCTS algorithm.
"""

from enum import Enum
from typing import Any, Callable, TypeVar
import numpy as np
import warnings
import functools


def randmax(A: Any) -> int:
    """
    Function to return the index of the element with highest value in A, breaking ties randomly.

    Args:
        A (Any): The array from which to select the index of the element with the highest value

    Examples:
        >>> node = MCTSNode(label=0, rank=2, n_children=2)
        >>> node.expand()
        >>> node[0].score = 0.5
        >>> randmax(node.get_children_scores())
        0
    """
    maxValue: Any = max(A)
    index: list[int] = [i for i in range(len(A)) if A[i] == maxValue]
    return int(np.random.choice(index))


class NormOption(Enum):
    """
    Class to represent the different normalization options for the MCTS algorithm
    """
    SOFTMAX = 1
    UNIFORM = 2
    NONE = 3


F = TypeVar('F', bound=Callable[..., Any])  # Used for type hinting


def debug(func: F) -> F:
    """
    Wrapper to help in debugging the MCTS algorithm
    Examples:
        >>> @debug
        ... def f(x: int) -> int:
        ...     return x
        >>> f(1)
        'f': args=(1,), kwargs={}
        'f': output=1
        1
    """
    def wrapper(*args, **kwargs) -> Any:
        print(f"'{func.__name__}': {args=}, {kwargs=}")
        output: Any = func(*args, **kwargs)
        print(f"'{func.__name__}': {output=}")
        return output
    return wrapper  # type: ignore


def deprecated(func: F) -> F:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func  # type: ignore


__all__: list[str] = ['randmax', 'NormOption', 'debug', 'deprecated']
