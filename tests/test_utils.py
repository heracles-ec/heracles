import numpy as np
import heracles


def test_add_to_cls():
    cls = {}
    cls[("P", "P", 1, 1)] = heracles.Result(np.ones(10))
    x = {}
    x[("P", "P", 1, 1)] = -1.0
    _cls = heracles.utils.add_to_Cls(cls, x)
    __cls = heracles.utils.sub_to_Cls(_cls, x)
    for key in list(cls.keys()):
        assert np.all(_cls[key] == np.zeros(10))
        assert np.all(cls[key].__array__() == __cls[key].__array__())


def test_get_cl():
    a = np.arange(10)
    ab = 2 * a
    ba = 3 * a
    cls = {}
    cls[("SHE", "POS", 1, 1)] = heracles.Result(np.array([a, a]), spin=(2, 0))
    cls[("SHE", "SHE", 2, 1)] = heracles.Result(
        np.array([[a, ab], [ba, a]]), spin=(2, 2)
    )

    cl = heracles.utils.get_cl(("POS", "SHE", 1, 1), cls)
    assert np.all(cl == np.array([a, a]))
    cl = heracles.utils.get_cl(("SHE", "SHE", 1, 2), cls)
    assert np.all(cl == np.array([[a, ba], [ab, a]]))


def test_expand_squeeze_spin0_dims(cls0, cov_jk):
    for key in list(cls0.keys()):
        cl = cls0[key]
        s1, s2 = cl.spin
        dof1 = 1 if s1 == 0 else 2
        dof2 = 1 if s2 == 0 else 2
        _cl = heracles.utils.expand_spin0_dims(cl)
        (_ax,) = _cl.axis
        assert _cl.shape == (dof1, dof2, cl.shape[-1])
        assert _ax == 2
        __cl = heracles.utils.squeeze_spin0_dims(_cl)
        assert np.all(cl.__array__() == __cl.__array__())
        assert cl.axis == __cl.axis

    for key in list(cov_jk.keys()):
        cov = cov_jk[key]
        sa1, sb1, sa2, sb2 = cov.spin
        dof_a1 = 1 if sa1 == 0 else 2
        dof_b1 = 1 if sb1 == 0 else 2
        dof_a2 = 1 if sa2 == 0 else 2
        dof_b2 = 1 if sb2 == 0 else 2
        _cov = heracles.utils.expand_spin0_dims(cov)
        _ax1, _ax2 = _cov.axis
        assert _cov.shape == (
            dof_a1,
            dof_b1,
            dof_a2,
            dof_b2,
            cov.shape[-2],
            cov.shape[-1],
        )
        assert (_ax1, _ax2) == (4, 5)
        __cov = heracles.utils.squeeze_spin0_dims(_cov)
        assert np.all(cov.__array__() == __cov.__array__())
        assert cov.axis == __cov.axis
