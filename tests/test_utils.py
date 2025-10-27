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


def test_get_cl():
    a = np.arange(10)
    ab = 2 * a
    ba = 3 * a
    cls = {}
    cls[("SHE", "POS", 1, 1)] = heracles.Result(np.array([a, a]), spin=(2, 0))
    cls[("SHE", "SHE", 2, 1)] = heracles.Result(np.array([[a, ab], [ba, a]]), spin=(2, 2))

    cl = dices.utils.get_cl(("POS", "SHE", 1, 1), cls)
    assert np.all(cl == np.array([a, a]))
    cl = dices.utils.get_cl(("SHE", "SHE", 1, 2), cls)
    assert np.all(cl == np.array([[a, ba], [ab, a]]))
