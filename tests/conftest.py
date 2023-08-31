import warnings
from contextlib import contextmanager

import pytest
from numba import config

config.DISABLE_JIT = True


@contextmanager
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
