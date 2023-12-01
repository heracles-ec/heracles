import numpy as np
import pytest
from numba import config

config.DISABLE_JIT = True


@pytest.fixture(scope="session")
def rng(seed: int = 50) -> np.random.Generator:
    return np.random.default_rng(seed)
