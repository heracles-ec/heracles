import numpy as np
import heracles
import heracles.dices as dices


def test_add_to_cls():
    cls = {}
    cls[("P", "P", 1, 1)] = heracles.Result(np.ones(10))
    x = {}
    x[("P", "P", 1, 1)] = -1.0
    _cls = dices.utils.add_to_Cls(cls, x)
    __cls = dices.utils.sub_to_Cls(_cls, x)
    for key in list(cls.keys()):
        assert np.all(_cls[key] == np.zeros(10))
        assert np.all(cls[key].__array__() == __cls[key].__array__())
