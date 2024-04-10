from unittest.mock import Mock, call, patch

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
        0: np.zeros(100),
        2: np.zeros(100, dtype=np.dtype(float, metadata={"bias": 4.56, "spin_2": 2})),
    }

    nbs = {
        0: 1.23,
    }

    debias_cls(cls, nbs, inplace=True)

    assert np.all(cls[0] == -1.23)

    assert np.all(cls[2][:2] == 0.0)
    assert np.all(cls[2][2:] == -4.56)


def test_debias_cls_healpix():
    import healpy as hp

    from heracles.twopoint import debias_cls

    pw0, pw2 = hp.pixwin(64, lmax=99, pol=True)

    md1 = {
        "kernel_1": "healpix",
        "nside_1": 64,
        "kernel_2": "healpix",
        "nside_2": 64,
        "spin_2": 2,
    }
    md2 = {
        "kernel_1": "healpix",
        "nside_1": 64,
        "spin_2": 2,
    }
    md3 = {
        "kernel_1": "healpix",
        "nside_1": 64,
        "kernel_2": "healpix",
        "nside_2": 64,
        "spin_2": 2,
        "deconv_2": False,
    }

    cls = {
        1: np.zeros(100, dtype=np.dtype(float, metadata=md1)),
        2: np.zeros(100, dtype=np.dtype(float, metadata=md2)),
        3: np.zeros(100, dtype=np.dtype(float, metadata=md3)),
    }

    nbs = {
        1: 1.23,
        2: 4.56,
        3: 7.89,
    }

    debias_cls(cls, nbs, inplace=True)

    assert np.all(cls[1][:2] == 0.0)
    assert np.all(cls[1][2:] == -1.23 / pw0[2:] / pw2[2:])

    assert np.all(cls[2][:2] == 0.0)
    assert np.all(cls[2][2:] == -4.56 / pw0[2:])

    assert np.all(cls[3][:2] == 0.0)
    assert np.all(cls[3][2:] == -7.89 / pw0[2:])


@patch("convolvecl.mixmat_eb")
@patch("convolvecl.mixmat")
def test_mixing_matrices(mock, mock_eb, rng):
    from heracles.twopoint import mixing_matrices

    # this only tests the function logic
    # the mixing matrix computation itself is tested elsewhere

    # field definition, requires mask and spin weight

    # mixmat_eb returns three values
    mock_eb.return_value = (Mock(), Mock(), Mock())

    lmax = 20
    cl = rng.standard_normal(lmax + 1)

    # create the mock field information
    fields = {
        "P": Mock(mask="V", spin=0),
        "G": Mock(mask="W", spin=2),
    }

    # compute pos-pos
    cls = {("V", "V", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 1
    assert mock.call_count == 1
    assert mock_eb.call_count == 0
    mock.assert_called_with(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0))
    assert mms["P", "P", 0, 1] is mock.return_value

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute pos-she
    cls = {("V", "W", 0, 1): cl, ("W", "V", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 2
    assert mock.call_count == 2
    assert mock_eb.call_count == 0
    assert mock.call_args_list == [
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 2)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(2, 0)),
    ]
    assert mms["P", "G_E", 0, 1] is mock.return_value
    assert mms["G_E", "P", 0, 1] is mock.return_value

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute she-she
    cls = {("W", "W", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 3
    assert mock.call_count == 0
    assert mock_eb.call_count == 1
    mock_eb.assert_called_with(cl, l1max=None, l2max=None, l3max=None, spin=(2, 2))
    assert mms["G_E", "G_E", 0, 1] is mock_eb.return_value[0]
    assert mms["G_B", "G_B", 0, 1] is mock_eb.return_value[1]
    assert mms["G_E", "G_B", 0, 1] is mock_eb.return_value[2]

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute unknown
    cls = {("X", "Y", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 0

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute multiple combinations
    cls = {("V", "V", 0, 0): cl, ("V", "V", 0, 1): cl, ("V", "V", 1, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 3
    assert mock.call_count == 3
    assert mock_eb.call_count == 0
    assert mock.call_args_list == [
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
    ]
    assert mms.keys() == {("P", "P", 0, 0), ("P", "P", 0, 1), ("P", "P", 1, 1)}


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
