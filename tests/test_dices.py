import numpy as np
import heracles
import pytest
import heracles.dices as dices

try:
    from copy import replace
except ImportError:
    # Python < 3.13
    from dataclasses import replace


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

    # Copy data map and add systematic map which should not be jackknifed
    data_maps_nojk = data_maps.copy()
    data_maps_nojk[("SYS", 1)] = np.arange(1, 11, dtype=float)

    # Copy Jackknife maps and add None map, output jackknifed maps
    jk_maps_nojk = jk_maps.copy()
    jk_maps_nojk[("SYS", 1)] = None
    out_maps = dices.jackknife.jackknife_maps(data_maps_nojk, jk_maps_nojk, jk=1)

    # Assert that the SYS map is unchanged
    np.testing.assert_allclose(out_maps[("SYS", 1)], data_maps_nojk[("SYS", 1)])

    # Check that a sample key WAS jackknifed
    sample_key = ("POS", 1)
    assert not np.allclose(out_maps[sample_key], data_maps_nojk[sample_key])


def test_cls(nside, cls0, fields, data_maps, vis_maps, jk_maps):
    _cls0 = dices.jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=0)[()]
    for key in list(_cls0.keys()):
        _cl = _cls0[key]
        *_, nells = _cl.shape
        assert nells == nside // 4 + 1
    for key in list(cls0.keys()):
        cl = cls0[key].array
        _cl = _cls0[key].array
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_bias(cls0):
    b = dices.jackknife.bias(cls0)
    for key in list(cls0.keys()):
        assert key in list(b.keys())


def test_get_delete1_fsky(jk_maps, njk):
    for jk in range(1, njk + 1):
        alphas = dices.jackknife_fsky(jk_maps, jk, jk, ratio=True)
        for key in list(alphas.keys()):
            _alpha = 1 - 1 / njk
            alpha = alphas[key]
            assert alpha == pytest.approx(_alpha, rel=1e-1)
        alphas = dices.jackknife_fsky(jk_maps, jk, jk, ratio=False)
        for key in list(alphas.keys()):
            _alpha = (njk - 1) / njk
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
            alphas = dices.jackknife_fsky(jk_maps, jk, jk2, ratio=False)
            for key in list(alphas.keys()):
                _alpha = (njk - 2) / njk
                alpha = alphas[key]
                assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_full_mask_correction(cls0, mls0, fields):
    alphas = dices.get_mask_correlation_ratio(mls0, mls0, unmixed=False)
    _cls = heracles.unmixing._natural_unmixing(cls0, alphas, fields)
    for key in list(cls0.keys()):
        cl = cls0[key].array
        _cl = _cls[key].array
        assert np.isclose(cl[2:], _cl[2:]).all()

    _alphas = dices.get_mask_correlation_ratio(mls0, mls0, unmixed=True)
    for key in list(_alphas.keys()):
        wmls0 = heracles.transforms._cl2corr(mls0[key]).T[0]
        alpha = alphas[key].array
        _alpha = _alphas[key].array / wmls0
        assert np.isclose(alpha, _alpha).all()


def test_fast_mask_correction(cls0, fields, jk_maps):
    _cls0 = dices.correct_footprint_reduction(cls0, jk_maps, fields, 0, 0)
    for key in list(cls0.keys()):
        cl = cls0[key].array
        _cl = _cls0[key].array
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_polspice(cls0):
    from heracles.utils import get_cl

    cls = np.array(
        [
            get_cl(("POS", "POS", 1, 1), cls0),
            get_cl(("SHE", "SHE", 1, 1), cls0)[0, 0],
            get_cl(("SHE", "SHE", 1, 1), cls0)[1, 1],
            get_cl(("POS", "SHE", 1, 1), cls0)[0],
        ]
    ).T
    corrs = heracles.transforms._cl2corr(cls)
    _cls = heracles.transforms._corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_jackknife(nside, njk, cov_jk, cls0, cls1):
    assert len(cls1) == njk
    for key in cls1.keys():
        cl = cls1[key]
        for key in list(cl.keys()):
            _cl = cl[key]
            *_, nells = _cl.shape
            assert nells == nside // 4 + 1

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
        assert (m, n) == (nside // 4 + 1, nside // 4 + 1)

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
    for key in list(debiased_cov.keys()):
        c = debiased_cov[key]
        _c = cov_jk[key]
        ell = c.shape[-1]
        # Create mask for off-diagonal elements
        offd_mask = ~np.eye(ell, dtype=bool)
        # Extract off-diagonal elements
        offd = c[..., offd_mask]
        _offd = _c[..., offd_mask]
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
    cov = {}
    # same as cov_jk but with rand entries
    for key in cov_jk.keys():
        g = cov_jk[key]
        s = g.shape
        cov[key] = heracles.Result(np.random.rand(*s), ell=g.ell, axis=g.axis)
    unit_matrix = {}
    for key in cov.keys():
        g = cov[key]
        s = g.shape
        *_, i = s
        single_diag = np.eye(i)
        # Expand to the desired shape using broadcasting
        a = np.broadcast_to(single_diag, s)
        unit_matrix[key] = replace(g, array=a)
    # Shrinkage factor
    # To do: is there a way of checking the shrinkage factor?
    shrinkage_factor = 0.5
    shrunk_cov = dices.shrink(unit_matrix, cov, shrinkage_factor)

    # Test that diagonals are not touched
    for key in list(shrunk_cov.keys()):
        c = shrunk_cov[key]
        _c = unit_matrix[key]
        c_diag = np.diagonal(c, axis1=-2, axis2=-1)
        _c_diag = np.diagonal(_c, axis1=-2, axis2=-1)
        assert np.allclose(c_diag, _c_diag, rtol=1e-5, atol=1e-5)


def test_flatten_cls(nside, cls0):
    from heracles.utils import _flatten, flatten

    # Check that the individual blocks are flattened correctly
    for key in cls0.keys():
        arr = cls0[key]
        *prefix, ell = arr.shape
        N = np.prod(prefix, dtype=int)
        flat = _flatten(arr)
        assert flat.shape == (N * ell)
        reconstructed = flat.reshape(N, ell).transpose(0, 1).reshape(*prefix, ell)
        assert np.allclose(arr.array, reconstructed)

    # Check flattened cls has correct shape
    _cls = flatten(cls0)
    assert len(_cls) == 30 * (nside // 4 + 1)


def test_flatten_cov(nside, cov_jk):
    from heracles.utils import _flatten, flatten

    # Check that the individual blocks are flattened correctly
    for key in cov_jk.keys():
        arr = cov_jk[key]
        *prefix, l1, l2 = arr.shape
        s1, s2, s3, s4 = arr.spin
        dof1 = 1 if s1 == 0 else 2
        dof2 = 1 if s2 == 0 else 2
        dof3 = 1 if s3 == 0 else 2
        dof4 = 1 if s4 == 0 else 2
        N1 = dof1 * dof2
        N2 = dof3 * dof4
        flat = _flatten(arr)
        assert flat.shape == (N1 * l1, N2 * l2)
        reconstructed = (
            flat.reshape(N1, l1, N2, l2).transpose(0, 2, 1, 3).reshape(*prefix, l1, l2)
        )
        assert np.allclose(arr.array, reconstructed)

    # Check flattened covariance has correct shape
    _cov = flatten(cov_jk)
    _, n = _cov.shape
    assert n == 30 * (nside // 4 + 1)


def test_gauss_cov(cls0, cov_jk):
    _cls0 = {}
    for key in list(cls0.keys()):
        a = cls0[key].array
        a = np.ones_like(a)
        _cls0[key] = replace(cls0[key], array=a)
    # We want to undo the bias that we will add later
    # for an easy check
    bias = dices.jackknife.bias(_cls0)
    _cls0 = heracles.utils.sub_to_Cls(_cls0, bias)

    # Compute Gaussian covariance
    gauss_cov = dices.gaussian_covariance(_cls0)

    # check for shape and keys
    for key in list(cov_jk.keys()):
        assert key in list(gauss_cov.keys())
        c1 = cov_jk[key]
        c2 = gauss_cov[key]
        assert c1.shape == c2.shape

    # check for diagonal values
    for key in list(gauss_cov.keys()):
        c = gauss_cov[key]
        c_diag = np.diagonal(c, axis1=-2, axis2=-1)
        _c_diag = 2 * np.ones_like(c_diag)
        assert np.allclose(c_diag, _c_diag, rtol=1e-5, atol=1e-5)
