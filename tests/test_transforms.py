import numpy as np
import heracles


def test_transforms(cls0):
    corr = heracles.cl2corr(cls0)
    for key in list(cls0.keys()):
        print(key, cls0[key].shape, corr[key].shape)
        assert corr[key].shape == cls0[key].shape

    _cls = heracles.corr2cl(corr)
    for key in list(cls0.keys()):
        print(key, cls0[key].shape, _cls[key].shape)
        assert _cls[key].shape == cls0[key].shape
        assert np.isclose(cls0[key].array, _cls[key].array).all()
