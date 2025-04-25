import numpy as np
import heracles
import pytest
from heracles.result import Result


def make_dummy_cls(x):
    cls = {
        ("POS", "POS", 1, 1): x,
        ("POS", "SHE", 1, 1): np.array([x, x]),
        ("SHE", "SHE", 1, 1): np.array([x, x, x]),
        ("POS", "POS", 1, 2): x,
        ("POS", "SHE", 1, 2): np.array([x, x]),
        ("POS", "SHE", 2, 1): np.array([x, x]),
        ("SHE", "SHE", 1, 2): np.array([x, x, x, x]),
        ("POS", "POS", 2, 2): x,
        ("POS", "SHE", 2, 2): np.array([x, x]),
        ("SHE", "SHE", 2, 2): np.array([x, x, x]),
    }
    for key in list(cls.keys()):
        cls[key] = Result(cls[key], axis=0, ell=np.arange(128))
    return cls


def make_dummy_mask(x):
    cls = {
        ("VIS", "VIS", 1, 1): x,
        ("VIS", "WHT", 1, 1): x,
        ("WHT", "WHT", 1, 1): x,
        ("VIS", "VIS", 1, 2): x,
        ("VIS", "WHT", 1, 2): x,
        ("VIS", "WHT", 2, 1): x,
        ("WHT", "WHT", 1, 2): x,
        ("VIS", "VIS", 2, 2): x,
        ("VIS", "WHT", 2, 2): x,
        ("WHT", "WHT", 2, 2): x,
    }
    for key in list(cls.keys()):
        cls[key] = Result(cls[key], axis=0, ell=np.arange(128))
    return cls


def make_dummy_m(x):
    y = np.diag(x)
    M = {
        ("POS", "POS", 1, 1): y,
        ("POS", "SHE", 1, 1): y,
        ("SHE", "SHE", 1, 1): np.array([y / 2, y / 2, y]),
        ("POS", "POS", 1, 2): y,
        ("POS", "SHE", 1, 2): y,
        ("POS", "SHE", 2, 1): y,
        ("SHE", "SHE", 1, 2): np.array([y / 2, y / 2, y]),
        ("POS", "POS", 2, 2): y,
        ("POS", "SHE", 2, 2): y,
        ("SHE", "SHE", 2, 2): np.array([y / 2, y / 2, y]),
    }
    for key in list(M.keys()):
        M[key] = Result(M[key], axis=0, ell=np.arange(128))
    return M


def test_forwards():
    x = np.random.rand(10)
    cls = make_dummy_cls(x)
    M = make_dummy_m(1 / x)
    _cls = heracles.forwards(cls, M)
    for key in list(cls.keys()):
        assert _cls[key].shape == cls[key].shape
    for key in list(cls.keys()):
        assert (_cls[key].array == np.ones(10)).all


def test_inversion():
    x = np.random.rand(10)
    cls = make_dummy_cls(x)
    M = make_dummy_m(x)
    _cls = heracles.inversion(cls, M)
    for key in list(cls.keys()):
        assert _cls[key].shape == cls[key].shape
    for key in list(cls.keys()):
        print(key, _cls[key].array, np.ones(10))
        assert (_cls[key].array == np.ones(10)).all


def test_natural_unmixing():
    x = np.random.rand(10)
    cls = make_dummy_cls(x)
    mask_cls = make_dummy_mask(x)
    _cls = heracles.natural_unmixing(cls, mask_cls, patch_hole=False)
    for key in list(cls.keys()):
        assert _cls[key].shape == cls[key].shape
    for key in list(cls.keys()):
        assert (_cls[key].array == np.ones(10)).all
