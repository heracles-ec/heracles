import numpy as np
import heracles
<<<<<<< HEAD
import pytest
import yaml
import heracles.dices as dices
=======
import dices
>>>>>>> 6596cde87ca2f08afe99d7e7d5d7a66a9279acb1


def test_add_to_cls():
    cls = {}
    cls[("P", "P", 1, 1)] = heracles.Result(np.ones(10))
    x = {}
    x[("P", "P", 1, 1)] = -1.0
    _cls = dices.add_to_Cls(cls, x)
    __cls = dices.sub_to_Cls(_cls, x)
    for key in list(cls.keys()):
        assert np.all(_cls[key] == np.zeros(10))
        assert np.all(cls[key].__array__() == __cls[key].__array__())


def test_cov2corr():
    cov = np.random.rand(10, 10)
    corr = dices.cov2corr(cov)
    diag = np.diag(corr)
    assert np.all(np.isclose(diag, np.ones(10)))
