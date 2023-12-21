import numpy as np
import pytest


@pytest.fixture
def nside():
    return 32


@pytest.fixture
def zbins():
    return {0: (0.0, 0.8), 1: (1.0, 1.2)}


@pytest.fixture
def mock_alms(rng, zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    names = ["P", "G_E", "G_B"]

    alms = {}
    for n in names:
        for i in zbins:
            a = rng.standard_normal((Nlm, 2)) @ [1, 1j]
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

    # explicit cross with separate alms

    mock_alms1 = {(k, i): alm for (k, i), alm in mock_alms.items() if i % 2 == 0}
    mock_alms2 = {(k, i): alm for (k, i), alm in mock_alms.items() if i % 2 == 1}

    order = ["P", "G_E", "G_B"]

    comb12 = {
        (k1, k2, i1, i2) if order.index(k1) <= order.index(k2) else (k2, k1, i2, i1)
        for k1, i1 in mock_alms1.keys()
        for k2, i2 in mock_alms2.keys()
    }

    cls = angular_power_spectra(mock_alms1, mock_alms2)

    assert cls.keys() == comb12


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


def test_mixing_matrices(rng):
    from heracles.twopoint import mixing_matrices

    # this only tests the function logic
    # the mixing matrix computation itself is tested elsewhere

    lmax = 20
    cl = rng.standard_normal(lmax + 1)

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


@pytest.mark.parametrize("weights", [None, "l(l+1)", "2l+1", "<rand>"])
def test_binned_cls(rng, weights):
    from heracles.twopoint import binned_cls

    cls = {"key": rng.standard_normal(21)}

    bins = [2, 5, 10, 15, 20]

    if weights == "<rand>":
        weights_ = rng.random(40)
    else:
        weights_ = weights

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


@pytest.mark.parametrize("weights", [None, "l(l+1)", "2l+1", "<rand>"])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_bin2pt(ndim, rng, weights):
    from heracles.twopoint import bin2pt

    data = rng.standard_normal((21, 31, 41)[:ndim])

    bins = [2, 5, 10, 15, 20]

    if weights == "<rand>":
        weights_ = rng.random(51)
    else:
        weights_ = weights

    result = bin2pt(data, bins, "XY", weights=weights_)

    ell = np.arange(len(data))

    if weights is None:
        w = np.ones_like(ell)
    elif weights == "l(l+1)":
        w = ell * (ell + 1)
    elif weights == "2l+1":
        w = 2 * ell + 1
    else:
        w = weights_[: len(ell)]

    binned_ell = np.empty(len(bins) - 1)
    binned_xy = np.empty((len(bins) - 1, *data.shape[1:]))
    binned_w = np.empty(len(bins) - 1)
    for i, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
        inbin = (a <= ell) & (ell < b)
        binned_ell[i] = np.average(ell[inbin], weights=w[inbin])
        for j in np.ndindex(*binned_xy.shape[1:]):
            binned_xy[(i, *j)] = np.average(data[(inbin, *j)], weights=w[inbin])
        binned_w[i] = w[inbin].sum()

    np.testing.assert_array_almost_equal(result["L"], binned_ell)
    np.testing.assert_array_almost_equal(result["XY"], binned_xy)
    np.testing.assert_array_equal(result["LMIN"], bins[:-1])
    np.testing.assert_array_equal(result["LMAX"], bins[1:])
    np.testing.assert_array_almost_equal(result["W"], binned_w)
