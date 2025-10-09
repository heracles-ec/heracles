import numpy as np
import heracles
import pytest
import heracles.dices as dices


def test_jkmap(jk_maps, njk):
    for key in list(jk_maps.keys()):
        assert np.all(np.unique(jk_maps[key]) == np.arange(1, njk + 1))


def test_jackknife_maps(data_maps, jk_maps, njk):
    # multiply maps by jk footprint
    vmap = np.copy(jk_maps[("VIS", 1)])
    vmap[vmap > 0] = vmap[vmap > 0] / vmap[vmap > 0]
    for key in list(data_maps.keys()):
        data_maps[key] *= vmap
    # test null case
    _data_maps = dices.jackknife.jackknife_maps(data_maps, jk_maps)
    for key in list(_data_maps.keys()):
        np.testing.assert_allclose(_data_maps[key], data_maps[key])
    # test delete1 case
    __data_maps = np.array(
        [
            dices.jackknife.jackknife_maps(data_maps, jk_maps, jk=i, jk2=i)[("POS", 1)]
            for i in range(1, njk + 1)
        ]
    )
    __data_map = np.sum(__data_maps, axis=0) / (njk - 1)
    np.testing.assert_allclose(__data_map, data_maps[("POS", 1)])
    ___data_map = np.prod(__data_maps, axis=0)
    np.testing.assert_allclose(___data_map, np.zeros_like(data_maps[("POS", 1)]))


def test_cls(nside, cls0, fields, data_maps, vis_maps, jk_maps):
    _cls0 = dices.jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=0)[()]
    for key in list(_cls0.keys()):
        _cl = _cls0[key]
        *_, nells = _cl.shape
        assert nells == nside//4 + 1
    for key in list(cls0.keys()):
        cl = cls0[key].__array__()
        _cl = _cls0[key].__array__()
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_bias(cls0):
    b = dices.jackknife.bias(cls0)
    for key in list(cls0.keys()):
        assert key in list(b.keys())


def test_get_delete1_fsky(jk_maps, njk):
    for jk in range(1, njk + 1):
        alphas = dices.jackknife_fsky(jk_maps, jk, jk)
        for key in list(alphas.keys()):
            _alpha = 1 - 1 / njk
            alpha = alphas[key]
            assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_get_delete2_fsky(jk_maps, njk):
    for jk in range(1, njk + 1):
        for jk2 in range(jk + 1, njk + 1):
            alphas = dices.jackknife_fsky(jk_maps, jk, jk2)
            for key in list(alphas.keys()):
                _alpha = 1 - 2 / njk
                alpha = alphas[key]
                assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_mask_correction(cls0, mls0):
    alphas = dices.mask_correction(mls0, mls0)
    _cls = heracles.unmixing._natural_unmixing(cls0, alphas)
    for key in list(cls0.keys()):
        cl = cls0[key].__array__()
        _cl = _cls[key].__array__()
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_polspice(cls0):
    cls = np.array(
        [
            cls0[("POS", "POS", 1, 1)],
            cls0[("SHE", "SHE", 1, 1)][0, 0],
            cls0[("SHE", "SHE", 1, 1)][1, 1],
            cls0[("POS", "SHE", 1, 1)][0],
        ]
    ).T
    corrs = heracles.cl2corr(cls)
    _cls = heracles.corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_jackknife(nside, njk, cov_jk, cls0, cls1):
    assert len(cls1) == njk
    for key in cls1.keys():
        cl = cls1[key]
        for key in list(cl.keys()):
            _cl = cl[key]
            *_, nells = _cl.shape
            assert nells == nside//4 + 1

    # Check correct number of delete1 cls
    assert len(list(cls1.keys())) == njk

    # Check for correct keys)
    cls_keys = list(cls0.keys())
    for i in range(0, len(cls_keys)):
        for j in range(i, len(cls_keys)):
            ki = cls_keys[i]
            kj = cls_keys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            _covkey = (A, B, C, D, nA, nB, nC, nD)
            assert _covkey in list(cov_jk.keys())

    # Check for correct shape
    for key in list(cov_jk.keys()):
        cov = cov_jk[key]
        *_, m, n = cov.shape
        assert (m, n) == (nside//4 + 1, nside//4 + 1)

    # re-arrange cqs1
    _cls1 = {}
    for k1 in cls0.keys():
        _cls1[k1] = [cls1[k2][k1].array for k2 in cls1.keys()]

    # Check against sample covariance
    for key in list(cls0.keys()):
        a, b, i, j = key
        cov_key = (a, b, a, b, i, j, i, j)
        cov = cov_jk[cov_key].array
        _cq = np.array(_cls1[key]).T
        prefactor = (njk - 1) ** 2 / (njk)
        print(f"Checking {key} with prefactor {prefactor}")
        if a == b == "POS":
            _cov = prefactor * np.cov(_cq)
            print(key)
            print((cov - _cov) / _cov)
            assert np.allclose(cov, _cov)
        elif a == b == "SHE":
            cov_E = cov[0, 0, 0, 0]
            cov_B = cov[1, 1, 1, 1]
            _cq_E = _cq[:, 0, 0]
            _cq_B = _cq[:, 1, 1]
            _cov_E = prefactor * np.cov(_cq_E)
            _cov_B = prefactor * np.cov(_cq_B)
            assert np.allclose(cov_E, _cov_E)
            assert np.allclose(cov_B, _cov_B)
        elif a == "POS" and b == "SHE":
            cov_E = cov[0, 0]
            cov_B = cov[1, 1]
            _cq_E = _cq[:, 0]
            _cq_B = _cq[:, 1]
            _cov_E = prefactor * np.cov(_cq_E)
            _cov_B = prefactor * np.cov(_cq_B)
            assert np.allclose(cov_E, _cov_E)
            assert np.allclose(cov_B, _cov_B)


def test_debiasing(cov_jk, cls0, cls1, cls2):
    # Debias
    debiased_cov = dices.debias_covariance(cov_jk, cls0, cls1, cls2)
    Q = dices.delete2_correction(
        cls0,
        cls1,
        cls2,
    )
    _debiased_cov = {}
    for key in list(cov_jk.keys()):
        _debiased_cov[key] = cov_jk[key].array - Q[key]

    # Check diagonal
    for key in list(debiased_cov.keys()):
        assert (debiased_cov[key] == _debiased_cov[key]).all()

    # Check off-diagonal
    for key in list(dices.io._fields2components(debiased_cov).keys()):
        c = dices.io._fields2components(debiased_cov)[key]
        _c = dices.io._fields2components(cov_jk)[key]
        offd_mask = ~np.eye(c.shape[0], dtype=bool)
        # Extract off-diagonal elements
        offd = c[offd_mask]
        _offd = _c[offd_mask]
        print(key, offd, _offd)
        assert np.allclose(offd, _offd)

    # Check keys
    keys1 = set(cov_jk.keys())
    keys2 = set(debiased_cov.keys())
    assert keys1 == keys2

    # Check for correct shape
    for key in list(cov_jk.keys()):
        C1 = cov_jk[key]
        C2 = debiased_cov[key]
        assert C1.shape == C2.shape


def test_shrinkage(cov_jk):
    # Fake target
    unit_matrix = {}
    for key in cov_jk.keys():
        g = cov_jk[key]
        s = g.shape
        *_, i = s
        single_diag = np.eye(i)  # Shape: (i, j)
        # Expand to the desired shape using broadcasting
        a = np.broadcast_to(single_diag, s)
        unit_matrix[key] = heracles.Result(a, ell=g.ell, axis=g.axis)
    # Shrinkage factor
    # To do: is there a way of checking the shrinkage factor?
    shrinkage_factor = 0.5
    shrunk_cov = dices.shrink(unit_matrix, cov_jk, shrinkage_factor)
    # Test that diagonals are not touched
    for key in list(shrunk_cov.keys()):
        c = shrunk_cov[key]
        _c = unit_matrix[key]
        c_diag = np.diagonal(c, axis1=-2, axis2=-1)
        _c_diag = np.diagonal(_c, axis1=-2, axis2=-1)
        c_diag = np.nan_to_num(c_diag)
        _c_diag = np.nan_to_num(_c_diag)
        print(key, c_diag, _c_diag)
        assert np.allclose(c_diag, _c_diag, rtol=1e-5, atol=1e-5)


def test_flatten(nside, cls0):
    lbins = 2
    ledges = np.logspace(np.log10(10), np.log10(nside//4), lbins + 1)
    cqs0 = heracles.binned(cls0, ledges)
    comp_cqs0 = dices.io._fields2components(cqs0)
    order = list(comp_cqs0.keys())
    cov = dices.gaussian_covariance(cqs0)
    # Flatten
    _cqs0 = dices.flatten(cqs0, order=order)
    __cqs0 = np.array([comp_cqs0[key].array for key in order]).flatten()
    flat_cov = dices.flatten(cov, order=order)
    (n,) = _cqs0.shape
    _n, _m = flat_cov.shape
    assert n == _n
    assert n == _m
    assert (_cqs0 == __cqs0).all()
    d_flat_cov = np.diag(flat_cov)
    _d_flat_cov = []
    comp_cov = dices.io._fields2components(cov)
    for key in order:
        a, b, i, j = key
        cov_key = (a, b, a, b, i, j, i, j)
        c = comp_cov[cov_key]
        d = np.diag(c)
        _d_flat_cov.append(d)
    _d_flat_cov = np.array(_d_flat_cov).flatten()
    assert d_flat_cov.shape == _d_flat_cov.shape
    for i, o in enumerate(order):
        print(
            o,
            d_flat_cov[i * lbins : (1 + i) * lbins],
            _d_flat_cov[i * lbins : (i + 1) * lbins],
        )
    assert (_d_flat_cov == d_flat_cov).all()


def test_gauss_cov(nside, cls0, cls1):
    lbins = 3
    ledges = np.logspace(np.log10(10), np.log10(nside//4), lbins + 1)
    cqs1 = heracles.binned(cls1, ledges)
    cqs0 = heracles.binned(cls0, ledges)
    cov_jk = dices.jackknife_covariance(cqs1)
    gauss_cov = dices.gaussian_covariance(cqs0)
    # Add bias
    b = dices.jackknife.bias(cqs0)
    cqs0 = dices.utils.add_to_Cls(cqs0, b)
    # Comp separate
    _cov_jk = dices.io._fields2components(cov_jk)
    _gauss_cov = dices.io._fields2components(gauss_cov)
    _cqs0 = dices.io._fields2components(cqs0)
    assert sorted(list(_cov_jk.keys())) == sorted(list(_gauss_cov.keys()))
    for key in list(_gauss_cov.keys()):
        a1, b1, a2, b2, i1, j1, i2, j2 = key
        key1 = a1, b1, i1, j1
        key2 = a2, b2, i2, j2
        if (key1 == key2) and ((a1, i1) == (b1, j1)) and ((a2, i2) == (b2, j2)):
            g = 2 * _cqs0[key1].array ** 2
            _g = dices.shrinkage._gaussian_covariance(_cqs0, key)
            __g = np.diag(_gauss_cov[key].array)
            assert (g == _g).all()
            assert (g == __g).all()
