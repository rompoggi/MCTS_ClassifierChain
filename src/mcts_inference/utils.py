"""
Utility functions and classes for the MCTS algorithm.

Debug is a wrapper to help in debugging any function

NormOption is an enum to represent the different normalization options for the MCTS algorithm.
"""

from enum import Enum
from typing import Any, Callable


class NormOption(Enum):
    """
    Class to represent the different normalization options for the MCTS algorithm
    """
    SOFTMAX = 1
    UNIFORM = 2
    NONE = 3


def debug(func) -> Callable[..., Any]:
    """
    Wrapper to help in debugging the MCTS algorithm

    Args:
        func (Callable[..., Any]): The function to wrap

    Returns:
        Callable[..., Any]: The wrapped function

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
    return wrapper


__all__: list[str] = ['NormOption', 'debug']
