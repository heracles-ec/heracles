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
    lmax = 128
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
    lmax = 128
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


def test_cls(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    jk_maps = make_jkmaps(data_path)
    fields = get_fields()
    data_cls = dices.jackknife.get_cls(data_maps, jk_maps, fields)
    _data_cls = dices.jackknife_cls(data_maps, vis_maps, jk_maps, fields, nd=0)[()]
    for key in list(data_cls.keys()):
        _cl = np.atleast_2d(data_cls[key])
        _, nells = _cl.shape
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
    _cls = dices.correct_mask(cls, mls, mls)
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
            cls[("SHE", "SHE", 1, 1)][0],
            cls[("SHE", "SHE", 1, 1)][1],
            cls[("POS", "SHE", 1, 1)][0],
        ]
    ).T
    corrs = dices.cl2corr(cls)
    _cls = dices.corr2cl(corrs)
    for cl, _cl in zip(cls.T, _cls.T):
        assert np.isclose(cl[2:], _cl[2:]).all()


def test_dices(data_path):
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
            _cl = np.atleast_2d(cl[key])
            ncls, nells = _cl.shape
            assert nells == nside + 1

    delete2_data_cls = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=2)
    assert len(delete2_data_cls) == 2 * JackNjk
    for jk in range(1, JackNjk + 1):
        for jk2 in range(jk + 1, JackNjk + 1):
            cl = delete2_data_cls[(jk, jk2)]
            for key in list(cl.keys()):
                _cl = np.atleast_2d(cl[key])
                _, nells = _cl.shape
                assert nells == nside + 1

    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    lgrid = (ledges[1:] + ledges[:-1]) / 2
    cqs0 = heracles.binned(data_cls, ledges)
    for key in list(cqs0.keys()):
        cq = np.atleast_2d(cqs0[key])
        ncls, nells = cq.shape
        assert nells == len(lgrid)
    cqs1 = heracles.binned(delete1_data_cls, ledges)
    for key in list(cqs1.keys()):
        for k in list(cqs1[key].keys()):
            cq = np.atleast_2d(cqs1[key][k])
            ncls, nells = cq.shape
            assert nells == len(lgrid)
    cqs2 = heracles.binned(delete2_data_cls, ledges)
    for key in list(cqs2.keys()):
        for k in list(cqs2[key].keys()):
            cq = np.atleast_2d(cqs2[key][k])
            ncls, nells = cq.shape
            assert nells == len(lgrid)

    # Delete1
    cov_jk = dices.jackknife_covariance(cqs1)
    #  Shrinkage
    # target_cov = dices.gaussian_covariance(cqs0)
    # shrinkage = dices.shrinkage_factor(cqs0, cqs1, target_cov)
    # shrunk_cov = dices.shrink_covariance(cqs0, cov_jk, target_cov, shrinkage)

    # Check for correct keys)
    cls_keys = list(cqs0.keys())
    k = 0
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
        _, _, m, n = cov.shape
        assert (m, n) == (lbins, lbins)

    debiased_cov = dices.debias_covariance(cov_jk, cqs0, cqs1, cqs2)
    Q = dices.delete2_correction(
        cqs0,
        cqs1,
        cqs2,
    )
    _debiased_cov = {}
    for key in list(cov_jk.keys()):
        _debiased_cov[key] = cov_jk[key].array - Q[key].array

    for key in list(debiased_cov.keys()):
        assert (debiased_cov[key] == _debiased_cov[key]).all()

    # Check keys
    keys1 = set(cov_jk.keys())
    keys2 = set(debiased_cov.keys())
    assert keys1 == keys2

    # Check for correct shape
    for key in list(cov_jk.keys()):
        C1 = cov_jk[key]
        C2 = debiased_cov[key]
        assert C1.shape == C2.shape

    # dices_cov = dices.dices_covariance(cqs0, shrunk_cov, debiased_cov)
    # _cqs0 = dices.Fields2Components(cqs0)
    # _cov1 = dices.Components2Data(_cqs0, shrunk_cov)
    # _cov2 = dices.Components2Data(_cqs0, debiased_cov)
    # _corr1 = dices.cov2corr(_cov1)
    # _var1 = np.diag(_cov1).copy()
    # _var2 = np.diag(_cov2).copy()
    # cond = np.where(_var2 < 0)[0]
    # _var2[cond] = _var1[cond]
    # _sig2 = np.sqrt(_var2)
    # _corr2 = np.outer(_sig2, _sig2)
    # _D = _corr2 * _corr1
    # _dices_cov = dices.Data2Components(_cqs0, _D)
    # for key in list(dices_cov.keys()):
    #     print(key)
    #     d = dices_cov[key]
    #     _d = _dices_cov[key]
    #     assert np.all(d == _d)


def test_cls_io(data_path):
    data_maps = make_data_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    _cls = dices.Fields2Components(cls)
    __cls = dices.Components2Fields(_cls)
    assert sorted(list(cls.keys())) == list(__cls.keys())
    for key in list(cls.keys()):
        cl = cls[key].array
        __cl = __cls[key].array
        assert (cl == __cl).all()


def test_cov_io(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls1 = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    cqs1 = heracles.binned(cls1, ledges)
    cov_jk = dices.jackknife_covariance(cqs1)
    _cov_jk = dices.Fields2Components(cov_jk)
    __cov_jk = dices.Components2Fields(_cov_jk)
    assert sorted(list(cov_jk.keys())) == sorted(list(__cov_jk.keys()))
    for key in list(cov_jk.keys()):
        cov = cov_jk[key].array
        __cov = __cov_jk[key].array
        assert (cov == __cov).all()


def test_data_io(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls0 = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    cls1 = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    cqs0 = heracles.binned(cls0, ledges)
    cqs1 = heracles.binned(cls1, ledges)
    cov_jk = dices.jackknife_covariance(cqs1)
    _cov_jk = dices.Fields2Components(cov_jk)
    __cov_jk = dices.Components2Data(_cov_jk)
    _cqs0 = dices.Fields2Components(cqs0)
    __cqs0 = dices.Components2Data(_cqs0)
    (n,) = __cqs0.shape
    _n, _m = __cov_jk.shape
    assert n == _n
    assert n == _m


def test_gauss_cov(data_path):
    nside = 128
    data_maps = make_data_maps()
    vis_maps = make_vis_maps()
    fields = get_fields()
    jkmaps = make_jkmaps(data_path)
    cls0 = dices.jackknife.get_cls(data_maps, jkmaps, fields)
    cls1 = dices.jackknife_cls(data_maps, vis_maps, jkmaps, fields, nd=1)
    lbins = 5
    ledges = np.logspace(np.log10(10), np.log10(nside), lbins + 1)
    cqs1 = heracles.binned(cls1, ledges)
    cqs0 = heracles.binned(cls0, ledges)
    cov_jk = dices.jackknife_covariance(cqs1)
    gauss_cov = dices.gaussian_covariance(cqs0)
    # Comp separate
    _cov_jk = dices.Fields2Components(cov_jk)
    _gauss_cov = dices.Fields2Components(gauss_cov)
    _cqs0 = dices.Fields2Components(cqs0)
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
