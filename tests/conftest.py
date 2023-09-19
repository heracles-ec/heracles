import warnings
from contextlib import contextmanager

import numpy as np
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


@pytest.fixture(scope="session")
def rng(seed: int = 50) -> np.random.Generator:
    return np.random.default_rng(seed)
