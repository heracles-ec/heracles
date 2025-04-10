import healpy as hp
import numpy as np
import heracles
import pytest
import heracles.dices as dices
from heracles.healpy import HealpixMapper
from heracles.result import Result
from heracles.fields import Positions, Shears, Visibility, Weights


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


def make_dummy_m(x):
    y = np.diag(x)
    M = {
        ("POS", "POS", 1, 1): y,
        ("POS", "SHE", 1, 1): y,
        ("SHE", "SHE", 1, 1): np.array([y/2, y/2, y]),
        ("POS", "POS", 1, 2): y,
        ("POS", "SHE", 1, 2): y,
        ("POS", "SHE", 2, 1): y,
        ("SHE", "SHE", 1, 2): np.array([y/2, y/2, y]),
        ("POS", "POS", 2, 2): y,
        ("POS", "SHE", 2, 2): y,
        ("SHE", "SHE", 2, 2): np.array([y/2, y/2, y]),
    }
    for key in list(M.keys()):
        M[key] = Result(M[key], axis=0, ell=np.arange(128))
    return M


def test_forwards():
    x = np.random.rand(128)
    cls = make_dummy_cls(x)
    M = make_dummy_m(1/x)
    fcls = heracles.forwards(cls, M)
    for key in list(cls.keys()):
        assert fcls[key].shape == cls[key].shape


def test_inversion():
    x = np.random.rand(128)
    cls = make_dummy_cls(x)
    M = make_dummy_m(x)
    fcls = heracles.inversion(cls, M)
    for key in list(cls.keys()):
        assert fcls[key].shape == cls[key].shape
