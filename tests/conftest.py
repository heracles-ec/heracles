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
def random_generator(random_seed: int = 50) -> np.random.Generator:
    """A generator object consistent across all tests

    Args:
        random_seed: A seed to initialise the BitGenerator

    Returns:
        The initialised generator object
    """
    return np.random.default_rng(random_seed)
