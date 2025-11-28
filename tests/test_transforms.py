import numpy as np
import heracles


def test_transforms(cls0):
    wcls0 = heracles.transforms.transform_cls(cls0)
    for key in list(cls0.keys()):
        cl = cls0[key]
        wcl = wcls0[key]
        assert wcl.array.shape == cl.array.shape
        assert cl.spin == wcl.spin
    _cls0 = heracles.transforms.transform_corrs(wcls0)
    for key in list(cls0.keys()):
        cl = cls0[key]
        _cl = _cls0[key]
        assert cl.spin == _cl.spin
        assert np.isclose(cl.array[2:], _cl.array[2:]).all()


def test_cl2corr_corr2cl():
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

    _cl = heracles.corr2cl(corr.T).T
    assert cl.shape == _cl.shape
    for c, _c in zip(cl, _cl):
        assert np.isclose(c[2:], _c[2:]).all()
