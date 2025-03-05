import healpy as hp
import numpy as np
import heracles
import pytest
import yaml
import heracles.dices as dices


def make_map():
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


def make_vis_map():
    nbins = 2
    nside = 128
    npix = hp.nside2npix(nside)
    map = 4 * np.ones(npix)
    maps = {}
    for i in range(1, nbins + 1):
        maps[("VIS", i)] = map
        maps[("WHT", i)] = np.array([map])
    return maps


def make_DICESObj(data_path):
    data_maps = make_map()
    vis_maps = make_vis_map()
    jkmap = hp.read_map(data_path / "jkmap.fits")
    with open(data_path / "params_test.yaml", mode="r") as file:
        config = yaml.safe_load(file)
    jkmaps = {}
    for key in list(vis_maps.keys()):
        jkmaps[key] = jkmap
    DICESObj = dices.DICES(data_maps, jkmaps, vis_maps, config)
    return DICESObj


def test_jkmap(data_path):
    DICESObj = make_DICESObj(data_path)
    Njk = DICESObj.JackNjk
    for key in list(DICESObj.jkmaps.keys()):
        assert np.all(np.unique(DICESObj.jkmaps[key]) == np.arange(0, Njk + 1))


def test_cls(data_path):
    DICESObj = make_DICESObj(data_path)
    data_cls, mask_cls = dices.get_cls(DICESObj.data_maps, DICESObj.vis_maps)
    for key in list(data_cls.keys()):
        _cl = np.atleast_2d(data_cls[key])
        _, nells = _cl.shape
        assert nells == DICESObj.nside + 1

    for key in list(mask_cls.keys()):
        _cl = np.atleast_2d(mask_cls[key])
        print(key, _cl.shape)
        _, nells = _cl.shape
        assert nells == DICESObj.nside + 1


def test_delete1_cls(data_path):
    DICESObj = make_DICESObj(data_path)
    delete1_data_cls, delete1_mask_cls = DICESObj.get_delete1_cls()
    assert len(delete1_data_cls) == DICESObj.JackNjk
    assert len(delete1_mask_cls) == DICESObj.JackNjk
    for key in delete1_mask_cls.keys():
        cl = delete1_data_cls[key]
        for key in list(cl.keys()):
            _cl = np.atleast_2d(cl[key])
            ncls, nells = _cl.shape
            assert nells == DICESObj.nside + 1
    for key in delete1_mask_cls.keys():
        cl = delete1_mask_cls[key]
        for key in list(cl.keys()):
            _cl = np.atleast_2d(cl[key])
            _, nells = _cl.shape
            assert nells == DICESObj.nside + 1


def test_delete2_cls(data_path):
    DICESObj = make_DICESObj(data_path)
    delete2_data_cls, delete2_mask_cls = DICESObj.get_delete2_cls()
    assert len(delete2_data_cls) == 2 * DICESObj.JackNjk
    assert len(delete2_mask_cls) == 2 * DICESObj.JackNjk
    for jk in range(1, DICESObj.JackNjk + 1):
        for jk2 in range(jk + 1, DICESObj.JackNjk + 1):
            cl = delete2_data_cls[(jk, jk2)]
            for key in list(cl.keys()):
                _cl = np.atleast_2d(cl[key])
                _, nells = _cl.shape
                assert nells == DICESObj.nside + 1
    for jk in range(1, DICESObj.JackNjk + 1):
        for jk2 in range(jk + 1, DICESObj.JackNjk + 1):
            cl = delete2_mask_cls[(jk, jk2)]
            for key in list(cl.keys()):
                _cl = np.atleast_2d(cl[key])
                _, nells = _cl.shape
                assert nells == DICESObj.nside + 1


def test_get_delete1_fsky(data_path):
    DICESObj = make_DICESObj(data_path)
    for jk in range(1, DICESObj.JackNjk + 1):
        alphas = dices.get_delete_fsky(DICESObj.jkmaps, jk, jk)
        for key in list(alphas.keys()):
            _alpha = 1 - 1 / DICESObj.JackNjk
            alpha = alphas[key]
            assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_get_delete2_fsky(data_path):
    DICESObj = make_DICESObj(data_path)
    for jk in range(1, DICESObj.JackNjk + 1):
        for jk2 in range(jk + 1, DICESObj.JackNjk + 1):
            alphas = dices.get_delete_fsky(DICESObj.jkmaps, jk, jk2)
            for key in list(alphas.keys()):
                _alpha = 1 - 2 / DICESObj.JackNjk
                alpha = alphas[key]
                assert alpha == pytest.approx(_alpha, rel=1e-1)


def test_delete1_cov(data_path):
    DICESObj = make_DICESObj(data_path)
    Cls0, _ = dices.get_cls(DICESObj.data_maps, DICESObj.vis_maps)
    Cqs0 = heracles.binned(Cls0, DICESObj.ledges)
    Clsjks, _ = DICESObj.get_delete1_cls()
    Cqsjks = heracles.binned(Clsjks, DICESObj.ledges)
    _, delete1_cov, _ = dices.get_delete1_cov(
        Cqs0,
        Cqsjks,
    )

    # Only test diagonal
    for key in list(delete1_cov.keys()):
        _k1 = (key[0], key[1], key[4], key[5])
        _k2 = (key[2], key[3], key[6], key[7])
        if _k1 != _k2:
            delete1_cov.pop(key)

    # Check for correct keys
    data_cls, _ = dices.get_cls(DICESObj.data_maps, DICESObj.vis_maps)
    compsep_cls = dices.compsep_Cls(data_cls)
    kk = list(delete1_cov.keys())
    k = list(compsep_cls.keys())
    for i in np.arange(len(kk)):
        _k1 = (kk[i][0], kk[i][1], kk[i][4], kk[i][5])
        _k2 = (kk[i][2], kk[i][3], kk[i][6], kk[i][7])
        if _k1 == _k2:
            assert _k1 == k[i]

    # Check for correct shape
    for key in list(delete1_cov.keys()):
        cov = delete1_cov[key]
        assert cov.shape == (DICESObj.lbins, DICESObj.lbins)


def test_delete2_cov(data_path):
    DICESObj = make_DICESObj(data_path)
    _, delete1_cov, _ = DICESObj.get_delete1_cov()
    delete2_cov = DICESObj.get_delete2_cov()

    Cls0, _ = dices.get_cls(DICESObj.data_maps, DICESObj.vis_maps)
    Clsjks, _ = DICESObj.get_delete1_cls()
    Clsjk2s, _ = DICESObj.get_delete2_cls()

    Cls0 = heracles.binned(Cls0, DICESObj.ledges)
    Clsjks = heracles.binned(Clsjks, DICESObj.ledges)
    Clsjk2s = heracles.binned(Clsjk2s, DICESObj.ledges)

    Q = dices.get_delete2_correction(
        Cls0,
        Clsjks,
        Clsjk2s,
    )
    _delete2_cov = dices.get_delete2_cov(delete1_cov, Q)

    for key in list(delete2_cov.keys()):
        assert (delete2_cov[key] == _delete2_cov[key]).all()


def test_dices_cov(data_path):
    DICESObj = make_DICESObj(data_path)
    Cls0, _ = dices.get_cls(DICESObj.data_maps, DICESObj.vis_maps)
    Cqs0 = heracles.binned(Cls0, DICESObj.ledges)
    Cqs0 = dices.compsep_Cls(Cqs0)
    cov1, _, T = DICESObj.get_delete1_cov()
    cov2 = DICESObj.get_delete2_cov()
    dices_cov = DICESObj.get_dices_cov()

    # Check keys
    keys1 = set(cov1.keys())
    keys2 = set(cov2.keys())
    keys3 = set(dices_cov.keys())
    assert keys1 == keys2 == keys3

    # Check for correct shape
    for key in list(cov1.keys()):
        C1 = cov1[key]
        C2 = cov2[key]
        CD = dices_cov[key]
        assert C1.shape == C2.shape == CD.shape

    # Check for delete2 correction
    _cov1 = dices.dict2mat(Cqs0, cov1)
    _cov2 = dices.dict2mat(Cqs0, cov2)
    _corr1 = dices.cov2corr(_cov1)
    _var1 = np.diag(_cov1).copy()
    _var2 = np.diag(_cov2).copy()
    cond = np.where(_var2 < 0)[0]
    _var2[cond] = _var1[cond]
    _sig2 = np.sqrt(_var2)
    _corr2 = np.outer(_sig2, _sig2)
    _D = _corr2 * _corr1
    _dices_cov = dices.mat2dict(Cqs0, _D)
    for key in list(dices_cov.keys()):
        assert np.all(dices_cov[key] == _dices_cov[key])
