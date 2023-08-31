import unittest.mock

import numpy as np
import pytest


@pytest.fixture()
def nside():
    return 32


@pytest.fixture()
def zbins():
    return {0: (0.0, 0.8), 1: (1.0, 1.2)}


@pytest.fixture()
def mock_alms(zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    names = ["P", "G_E", "G_B"]

    alms = {}
    for n in names:
        for i in zbins:
            a = np.random.randn(Nlm, 2) @ [1, 1j]
            a.dtype = np.dtype(a.dtype, metadata={"nside": 32})
            alms[n, i] = a

    return alms


def test_angular_power_spectra(mock_alms):
    from itertools import combinations_with_replacement

    from heracles.twopoint import angular_power_spectra

    # alms cross themselves

    comb = {
        (k1, k2, i1, i2)
        for (k1, i1), (k2, i2) in combinations_with_replacement(mock_alms, 2)
    }

    cls = angular_power_spectra(mock_alms)

    assert cls.keys() == comb

    # explicit cross

    cls = angular_power_spectra(mock_alms, mock_alms)

    assert cls.keys() == comb

    # explicit include

    cls = angular_power_spectra(
        mock_alms,
        include=[("P", "P", ..., ...), ("P", "G_E", ..., ...)],
    )

    assert cls.keys() == {
        (k1, k2, i1, i2)
        for k1, k2, i1, i2 in comb
        if (k1, k2) in [("P", "P"), ("P", "G_E")]
    }

    cls = angular_power_spectra(mock_alms, include=[("P", "P", 0), ("P", "G_E", 1)])

    assert cls.keys() == {
        (k1, k2, i1, i2)
        for k1, k2, i1, i2 in comb
        if (k1, k2, i1) in [("P", "P", 0), ("P", "G_E", 1)]
    }

    # explicit exclude

    cls = angular_power_spectra(
        mock_alms,
        exclude=[("P", "P"), ("P", "G_E"), ("P", "G_B")],
    )

    assert cls.keys() == {
        (k1, k2, i1, i2)
        for k1, k2, i1, i2 in comb
        if (k1, k2) not in [("P", "P"), ("P", "G_E"), ("P", "G_B")]
    }

    cls = angular_power_spectra(mock_alms, exclude=[(..., ..., 1, ...)])

    assert cls.keys() == {(k1, k2, i1, i2) for k1, k2, i1, i2 in comb if i1 != 1}


def test_debias_cls():
    from heracles.twopoint import debias_cls

    cls = {
        ("PP", 0, 0): np.zeros(100),
    }

    nbs = {
        ("PP", 0, 0): 1.23,
    }

    debias_cls(cls, nbs, inplace=True)

    assert np.all(cls["PP", 0, 0] == -1.23)


def test_mixing_matrices():
    from heracles.twopoint import mixing_matrices

    # this only tests the function logic
    # the mixing matrix computation itself is tested elsewhere

    lmax = 20
    cl = np.random.randn(lmax + 1)

    # compute pos-pos
    cls = {("V", "V", 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 1
    assert mms["00", 0, 1].shape == (lmax + 1, lmax + 1)

    # compute pos-she
    cls = {("V", "W", 0, 1): cl, ("W", "V", 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 2
    assert mms["0+", 0, 1].shape == (lmax + 1, lmax + 1)
    assert mms["0+", 1, 0].shape == (lmax + 1, lmax + 1)

    # compute she-she
    cls = {("W", "W", 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 3
    assert mms["++", 0, 1].shape == (lmax + 1, lmax + 1)
    assert mms["--", 0, 1].shape == (lmax + 1, lmax + 1)
    assert mms["+-", 0, 1].shape == (lmax + 1, lmax + 1)
    np.testing.assert_allclose(mms["+-", 0, 1], mms["++", 0, 1] + mms["--", 0, 1])

    # compute unknown
    cls = {("X", "Y", 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 1
    assert mms["XY", 0, 1].shape == (lmax + 1, lmax + 1)


def test_pixelate_mms_healpix():
    import healpy as hp

    from heracles.twopoint import pixelate_mms_healpix

    nside = 512
    lmax = 1000

    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)

    mms = {
        ("00", 0, 0): np.eye(lmax + 1),
        ("0+", 0, 0): np.eye(lmax + 1),
        ("++", 0, 0): np.eye(lmax + 1),
        ("--", 0, 0): np.eye(lmax + 1),
        ("+-", 0, 0): np.eye(lmax + 1),
        ("ab", 0, 0): np.eye(lmax + 1),
    }

    pixelate_mms_healpix(mms, nside, inplace=True)

    assert np.all(mms["00", 0, 0] == np.diag(fl0 * fl0))
    assert np.all(mms["0+", 0, 0] == np.diag(fl0 * fl2))
    assert np.all(mms["++", 0, 0] == np.diag(fl2 * fl2))
    assert np.all(mms["--", 0, 0] == np.diag(fl2 * fl2))
    assert np.all(mms["+-", 0, 0] == np.diag(fl2 * fl2))
    assert np.all(mms["ab", 0, 0] == np.diag(fl0 * fl0))


@pytest.mark.parametrize("weights", [None, "l(l+1)", "2l+1", "<rand>"])
def test_binned_cls(weights):
    from heracles.twopoint import binned_cls

    cls = {"key": np.random.randn(21)}

    bins = [2, 5, 10, 15, 20]

    weights_ = np.random.rand(40) if weights == "<rand>" else weights
    result = binned_cls(cls, bins, weights=weights_)

    for key, cl in cls.items():
        ell = np.arange(len(cl))

        if weights is None:
            w = np.ones_like(ell)
        elif weights == "l(l+1)":
            w = ell * (ell + 1)
        elif weights == "2l+1":
            w = 2 * ell + 1
        else:
            w = weights_[: len(ell)]

        binned_ell = []
        binned_cl = []
        binned_w = []
        for a, b in zip(bins[:-1], bins[1:]):
            inbin = (a <= ell) & (ell < b)
            binned_ell.append(np.average(ell[inbin], weights=w[inbin]))
            binned_cl.append(np.average(cl[inbin], weights=w[inbin]))
            binned_w.append(w[inbin].sum())

        np.testing.assert_array_almost_equal(result[key]["L"], binned_ell)
        np.testing.assert_array_almost_equal(result[key]["CL"], binned_cl)
        np.testing.assert_array_equal(result[key]["LMIN"], bins[:-1])
        np.testing.assert_array_equal(result[key]["LMAX"], bins[1:])
        np.testing.assert_array_almost_equal(result[key]["W"], binned_w)


@pytest.mark.parametrize("full", [False, True])
def test_random_noisebias(full):
    from heracles.twopoint import random_noisebias

    nside = 64
    npix = 12 * nside**2

    catalog = unittest.mock.Mock()
    catalog.visibility = None

    map_a = unittest.mock.Mock(side_effect=lambda _: np.random.rand(npix))
    map_b = unittest.mock.Mock(side_effect=lambda _: np.random.rand(npix))

    initial_randomize = [map_a.randomize, map_b.randomize]

    maps = {"A": map_a, "B": map_b}
    catalogs = {0: catalog, 1: catalog}

    nbs = random_noisebias(maps, catalogs, repeat=5, full=full)

    for m, r in zip(maps.values(), initial_randomize):
        assert m.randomize is r

    keys = [("A", "A", 0, 0), ("A", "A", 1, 1), ("B", "B", 0, 0), ("B", "B", 1, 1)]
    if full:
        keys += [
            ("A", "A", 0, 1),
            ("A", "B", 0, 0),
            ("A", "B", 1, 1),
            ("A", "B", 0, 1),
            ("A", "B", 1, 0),
            ("B", "B", 0, 1),
        ]

    assert set(nbs.keys()) == set(keys)

    # filter with include and exclude

    nbs = random_noisebias(
        maps,
        catalogs,
        repeat=5,
        full=full,
        include=[("A", 0), ("B", ...)],
        exclude=[("B", 0)],
    )

    keys = [("A", "A", 0, 0), ("B", "B", 1, 1)]
    if full:
        keys += [("A", "B", 0, 1)]

    assert set(nbs.keys()) == set(keys)
