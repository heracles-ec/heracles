import numpy as np
import heracles


def test_cl_transform(cls0):
    from heracles.utils import get_cl

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

    cls = np.array(
        [
            get_cl(("POS", "POS", 1, 1), cls0),
            get_cl(("SHE", "SHE", 1, 1), cls0)[0, 0],
            get_cl(("SHE", "SHE", 1, 1), cls0)[1, 1],
            get_cl(("POS", "SHE", 1, 1), cls0)[0],
        ]
    ).T
    corrs = heracles.cl2corr(cls)
    _cls = heracles.corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()
