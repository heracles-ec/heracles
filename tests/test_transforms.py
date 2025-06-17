import numpy as np
import heracles


def test_cl2corr():
    # Is there something more clever we can do here?
    # Like transforming the legendre nodes and return ones?
    cl = np.ones((4, 10))
    corr = heracles.cl2corr(cl.T).T
    assert corr.shape == cl.shape


def test_corr2cl():
    corr = np.ones((4, 10))
    cl = heracles.corr2cl(corr.T).T
    assert corr.shape == cl.shape


def test_transforms():
    cls = np.ones((4, 10)).T
    corrs = heracles.cl2corr(cls)
    _cls = heracles.corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()
