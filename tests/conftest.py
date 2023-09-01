import contextlib
import warnings

import numba
import pytest

numba.config.DISABLE_JIT = True


@contextlib.contextmanager
def warns(*types):
    if types == (None,):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                yield
            finally:
                pass
    else:
        with pytest.warns(*types):
            try:
                yield
            finally:
                pass
