import numpy as np
import pytest
from pathlib import Path
from numba import config

config.DISABLE_JIT = True


@pytest.fixture(scope="session")
def rng(seed: int = 50) -> np.random.Generator:
    return np.random.default_rng(seed)

@pytest.fixture
def data_path():
    return Path(__file__).parent / "data"