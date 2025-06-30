import numpy as np
import heracles


def test_cl2corr():
    # Is there something more clever we can do here?
    # Like transforming the legendre nodes and return ones?
    cl = np.array(
        [
            np.ones(10),
            np.ones(10),
            np.ones(10),
            np.zeros(10),
        ]
    )
    corr = heracles.cl2corr(cl.T).T
    assert corr.shape == cl.shape


def test_corr2cl():
    corr = np.array(
        [
            np.ones(10),
            np.zeros(10),
            np.zeros(10),
            np.zeros(10),
        ]
    )
    cl = heracles.corr2cl(corr.T).T
    assert corr.shape == cl.shape
