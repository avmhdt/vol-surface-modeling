"""Typed wrapper around Numba's ``njit``, which is untyped."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numba

F = TypeVar("F", bound=Callable[..., Any])


def njit(*, parallel: bool = False) -> Callable[[F], F]:
    """``numba.njit`` with the on-disk cache on, keeping the wrapped signature."""
    return cast(Callable[[F], F], numba.njit(cache=True, parallel=parallel))
