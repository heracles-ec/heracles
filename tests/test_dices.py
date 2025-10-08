import healpy as hp
import numpy as np
import heracles
import pytest
import heracles.dices as dices
from heracles.healpy import HealpixMapper
from heracles.fields import Positions, Shears, Visibility, Weights


def make_data_maps():
    nbins = 2
    nside = 128
    lmax = 20
    npix = hp.nside2npix(nside)
    fsky = 1 / 2
    ngal = 4.0
    wmean = 1.0
    var = 1.0
    bias = 4 * np.pi * fsky**2 * (var / wmean**2) / ngal
    map_p = 4 * np.ones(npix)
    nbar = (ngal * wmean) / (fsky * npix)
    heracles.update_metadata(
        map_p,
        nside=nside,
        lmax=lmax,
        ngal=ngal,
        nbar=nbar,
        wmean=wmean,
        bias=bias,
        var=var,
        variance=var / wmean**2,
        neff=ngal / (4 * np.pi * fsky),
        fsky=fsky,
        spin=0,
    )
    map_g = 4 * np.ones(npix)
    heracles.update_metadata(
        map_g,
        nside=nside,
        lmax=lmax,
        ngal=ngal,
        nbar=nbar,
        wmean=wmean,
        bias=bias,
        var=var,
        variance=var / wmean**2,
        neff=ngal / (2 * np.pi * fsky),
        fsky=fsky,
        spin=2,
    )
    maps = {}
    for i in range(1, nbins + 1):
        maps[("POS", i)] = map_p
        maps[("SHE", i)] = np.array([map_g, map_g])
    return maps


def make_vis_maps():
    nbins = 2
    nside = 128
    npix = hp.nside2npix(nside)
    map = 4 * np.ones(npix)
    maps = {}
    heracles.update_metadata(
        map,
        nside=nside,
        lmax=nside,
        bias=0.0,
        fsky=1 / 2,
        spin=0,
    )
    for i in range(1, nbins + 1):
        maps[("VIS", i)] = map
        maps[("WHT", i)] = np.array([map])
    return maps


def get_fields():
    """
    Internal method to initialize fields.
    inputs:
        nside (int): Healpix nside
        lmax (int): Maximum multipole
    returns:
        fields (dict): Dictionary of fields
    """
    nside = 128
    lmax = 20
    mapper = HealpixMapper(nside=nside, lmax=lmax)
    fields = {
        "POS": Positions(mapper, mask="VIS"),
        "SHE": Shears(mapper, mask="WHT"),
        "VIS": Visibility(mapper),
        "WHT": Weights(mapper),
    }
    return fields


def make_jkmaps(data_path):
    vis_maps = make_vis_maps()
    jkmap = hp.read_map(data_path / "jkmap.fits")
    jkmaps = {}
    for key in list(vis_maps.keys()):
        jkmaps[key] = jkmap
    return jkmaps


def test_jkmap(data_path):
    Njk = 5
    jkmaps = make_jkmaps(data_path)
    for key in list(jkmaps.keys()):
        assert np.all(np.unique(jkmaps[key]) == np.arange(0, Njk + 1))


def test_jackknife_maps(data_path):
    Njk = 5
    data_maps = make_data_maps()
    jk_maps = make_jkmaps(data_path)
    # multiply maps by jk footprint
    vmap = jk_maps[("VIS", 1)]
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
            for i in range(1, Njk + 1)
        ]
    )
    __data_map = np.sum(__data_maps, axis=0) / (Njk - 1)
    np.testing.assert_allclose(__data_map, data_maps[("POS", 1)])
    ___data_map = np.prod(__data_maps, axis=0)
    np.testing.assert_allclose(___data_map, np.zeros_like(data_maps[("POS", 1)]))


def test_cls(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    jk_maps = make_jkmaps(data_path)
    fields = get_fields()
    data_cls = dices.jackknife.get_cls(data_maps, jk_maps, fields)
    _data_cls = dices.jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=0)[()]
    for key in list(data_cls.keys()):
        _cl = data_cls[key]
        *_, nells = _cl.shape
        assert nells == nside + 1
    for key in list(data_cls.keys()):
        cl = data_cls[key].__array__()
        _cl = _data_cls[key].__array__()
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_bias(data_path):
    data_maps = make_data_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    b = dices.jackknife.bias(cls)
    for key in list(cls.keys()):
        assert key in list(b.keys())


def test_get_delete1_fsky(data_path):
    JackNjk = 5
    jkmaps = make_jkmaps(data_path)
    for jk in range(1, JackNjk + 1):
        alphas = dices.jackknife_fsky(jkmaps, jk, jk)
        for key in list(alphas.keys()):
            _alpha = 1 - 1 / JackNjk
            alpha = alphas[key]
            assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_get_delete2_fsky(data_path):
    JackNjk = 5
    jkmaps = make_jkmaps(data_path)
    for jk in range(1, JackNjk + 1):
        for jk2 in range(jk + 1, JackNjk + 1):
            alphas = dices.jackknife_fsky(jkmaps, jk, jk2)
            for key in list(alphas.keys()):
                _alpha = 1 - 2 / JackNjk
                alpha = alphas[key]
                assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_mask_correction(data_path):
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    mls = dices.jackknife.get_cls(vis_maps, jkmaps, fields)
    alphas = dices.mask_correction(mls, mls)
    _cls = heracles.unmixing._natural_unmixing(cls, alphas)
    for key in list(cls.keys()):
        cl = cls[key].__array__()
        _cl = _cls[key].__array__()
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_polspice(data_path):
    data_maps = make_data_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    cls = np.array(
        [
            cls[("POS", "POS", 1, 1)],
            cls[("SHE", "SHE", 1, 1)][0, 0],
            cls[("SHE", "SHE", 1, 1)][1, 1],
            cls[("POS", "SHE", 1, 1)][0],
        ]
    ).T
    corrs = heracles.cl2corr(cls)
    _cls = heracles.corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_jackknife(data_path):
    Njk = 5
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)

    cls0 = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    cls1 = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    assert len(cls1) == Njk
    for key in cls1.keys():
        cl = cls1[key]
        for key in list(cl.keys()):
            _cl = cl[key]
            *_, nells = _cl.shape
            assert nells == nside + 1

    # Check correct number of delete1 cls
    assert len(list(cls1.keys())) == Njk

    # Delete1
    cov_jk = dices.jackknife_covariance(cls1)

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
        assert (m, n) == (nside + 1, nside + 1)

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
        prefactor = (Njk - 1) ** 2 / (Njk)
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


def test_debiasing(data_path):
    JackNjk = 5
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)

    data_cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)

    delete1_data_cls = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    assert len(delete1_data_cls) == JackNjk
    for key in delete1_data_cls.keys():
        cl = delete1_data_cls[key]
        for key in list(cl.keys()):
            _cl = cl[key]
            *_, nells = _cl.shape
            assert nells == nside + 1

    delete2_data_cls = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=2)
    assert len(delete2_data_cls) == 2 * JackNjk
    for jk in range(1, JackNjk + 1):
        for jk2 in range(jk + 1, JackNjk + 1):
            cl = delete2_data_cls[(jk, jk2)]
            for key in list(cl.keys()):
                _cl = cl[key]
                *_, nells = _cl.shape
                assert nells == nside + 1

    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    cqs0 = heracles.binned(data_cls, ledges)
    for key in list(cqs0.keys()):
        cq = cqs0[key]
        *_, nells = cq.shape
        assert nells == len(lgrid)
    cqs1 = heracles.binned(delete1_data_cls, ledges)
    for key in list(cqs1.keys()):
        for k in list(cqs1[key].keys()):
            cq = cqs1[key][k]
            *_, nells = cq.shape
            assert nells == len(lgrid)
    cqs2 = heracles.binned(delete2_data_cls, ledges)
    for key in list(cqs2.keys()):
        for k in list(cqs2[key].keys()):
            cq = cqs2[key][k]
            *_, nells = cq.shape
            assert nells == len(lgrid)

    # Delete1
    cov_jk = dices.jackknife_covariance(cqs1)

    # Debias
    debiased_cov = dices.debias_covariance(cov_jk, cqs0, cqs1, cqs2)
    Q = dices.delete2_correction(
        cqs0,
        cqs1,
        cqs2,
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


def test_shrinkage(data_path):
    JackNjk = 5
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)

    data_cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)

    delete1_data_cls = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    assert len(delete1_data_cls) == JackNjk
    for key in delete1_data_cls.keys():
        cl = delete1_data_cls[key]
        for key in list(cl.keys()):
            _cl = cl[key]
            *_, nells = _cl.shape
            assert nells == nside + 1

    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    cqs0 = heracles.binned(data_cls, ledges)
    for key in list(cqs0.keys()):
        cq = cqs0[key]
        *_, nells = cq.shape
        assert nells == len(lgrid)
    cqs1 = heracles.binned(delete1_data_cls, ledges)
    for key in list(cqs1.keys()):
        for k in list(cqs1[key].keys()):
            cq = cqs1[key][k]
            *_, nells = cq.shape
            assert nells == len(lgrid)

    # Delete1
    cov_jk = dices.jackknife_covariance(cqs1)

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

    # Random matrix
    random_matrix = {}
    for key in cov_jk.keys():
        g = cov_jk[key]
        s = g.shape
        a = np.abs(np.random.rand(*s))
        random_matrix[key] = heracles.Result(a, ell=g.ell, axis=g.axis)

    # Shrinkage factor
    # To do: is there a way of checking the shrinkage factor?
    shrinkage_factor = dices.shrinkage_factor(cqs1, unit_matrix)

    # Check that the shrinkage factor is between 0 and 1
    assert 0 <= shrinkage_factor <= 1

    # Shrinkage
    shrunk_cov = dices.shrink(unit_matrix, random_matrix, shrinkage_factor)

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


def test_flatten(data_path):
    nside = 128
    lbins = 2
    data_maps = make_data_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls0 = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
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


def test_gauss_cov(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls0 = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    cls1 = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    lbins = 3
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
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
