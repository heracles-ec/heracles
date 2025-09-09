from unittest.mock import Mock, call, patch

import numpy as np
import pytest


@pytest.fixture
def nside():
    return 32


@pytest.fixture
def lmax():
    return 32


@pytest.fixture
def zbins():
    return {0: (0.0, 0.8), 1: (1.0, 1.2)}


@pytest.fixture
def mock_alms(rng, zbins, lmax):
    import numpy as np

    size = (lmax + 1) * (lmax + 2) // 2

    # names and spins
    fields = {"POS": 0, "SHE": 2}

    alms = {}
    for n, s in fields.items():
        shape = (size, 2) if s == 0 else (2, size, 2)
        for i in zbins:
            a = rng.standard_normal(shape) @ [1, 1j]
            a.dtype = np.dtype(a.dtype, metadata={"nside": 32, "spin": s})
            alms[n, i] = a

    return alms


def test_alm2lmax(rng):
    import healpy as hp

    from heracles.twopoint import alm2lmax

    for lmax in rng.choice(1000, size=100):
        alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
        assert alm2lmax(alm) == lmax


def test_alm2cl(mock_alms):
    from itertools import combinations_with_replacement

    import healpy as hp

    from heracles.twopoint import alm2cl

    for alm, alm2 in combinations_with_replacement(mock_alms.values(), 2):
        cl = alm2cl(alm, alm2)
        expected = np.empty_like(cl)
        for i in np.ndindex(*alm.shape[:-1]):
            for j in np.ndindex(*alm2.shape[:-1]):
                expected[i + j] = hp.alm2cl(alm[i], alm2[j])
        np.testing.assert_allclose(cl, expected)


def test_alm2cl_unequal_size(rng):
    import healpy as hp

    from heracles.twopoint import alm2cl

    lmax1 = 10
    lmax2 = 20

    alm = rng.standard_normal(((lmax1 + 1) * (lmax1 + 2) // 2, 2)) @ [1, 1j]
    alm2 = rng.standard_normal(((lmax2 + 1) * (lmax2 + 2) // 2, 2)) @ [1, 1j]

    alm21 = np.zeros_like(alm)
    for ell in range(lmax1 + 1):
        for m in range(ell + 1):
            alm21[hp.Alm.getidx(lmax1, ell, m)] = alm2[hp.Alm.getidx(lmax2, ell, m)]

    cl = alm2cl(alm, alm2)
    np.testing.assert_allclose(cl, hp.alm2cl(alm, alm21))

    cl = alm2cl(alm, alm2, lmax=lmax2)
    np.testing.assert_allclose(cl, hp.alm2cl(alm, alm21, lmax_out=lmax2))


def test_angular_power_spectra(mock_alms, lmax):
    from heracles.twopoint import angular_power_spectra

    # expected combinations of input alms and their shapes
    comb = {
        ("POS", "POS", 0, 0): (lmax + 1,),
        ("POS", "POS", 0, 1): (lmax + 1,),
        ("POS", "POS", 1, 1): (lmax + 1,),
        ("POS", "SHE", 0, 0): (2, lmax + 1),
        ("POS", "SHE", 0, 1): (2, lmax + 1),
        ("POS", "SHE", 1, 0): (2, lmax + 1),
        ("POS", "SHE", 1, 1): (2, lmax + 1),
        ("SHE", "SHE", 0, 0): (2, 2, lmax + 1),
        ("SHE", "SHE", 0, 1): (2, 2, lmax + 1),
        ("SHE", "SHE", 1, 1): (2, 2, lmax + 1),
    }

    # alms cross themselves
    cls = angular_power_spectra(mock_alms)
    keys = set(cls.keys())
    assert keys == comb.keys()
    for key, cl in cls.items():
        assert cl.shape == comb[key]
        assert cl.axis == (cl.ndim - 1,)

    # explicit cross
    cls = angular_power_spectra(mock_alms, mock_alms)
    keys = set(cls.keys())
    assert keys == comb.keys()
    for key, cl in cls.items():
        assert cl.shape == comb[key]

    # explicit cross with separate alms
    mock_alms1 = {(k, i): alm for (k, i), alm in mock_alms.items() if i % 2 == 0}
    mock_alms2 = {(k, i): alm for (k, i), alm in mock_alms.items() if i % 2 == 1}

    comb12 = {
        ("POS", "POS", 0, 1): (lmax + 1,),
        ("POS", "SHE", 0, 1): (2, lmax + 1),
        ("POS", "SHE", 1, 0): (2, lmax + 1),
        ("SHE", "SHE", 0, 1): (2, 2, lmax + 1),
    }

    cls = angular_power_spectra(mock_alms1, mock_alms2)
    keys = set(cls.keys())
    assert keys == comb12.keys()
    for key, cl in cls.items():
        assert cl.shape == comb12[key]

    # include and exclude
    inc = object()
    exc = object()
    with patch("heracles.twopoint.toc_match") as mock_match:
        mock_match.return_value = False
        cls = angular_power_spectra(mock_alms1, mock_alms2, include=inc, exclude=exc)
    assert len(cls) == 0
    assert mock_match.call_count == len(comb12)
    call_iter = iter(mock_match.call_args_list)
    for a, i in mock_alms1:
        for b, j in mock_alms2:
            assert next(call_iter) == call((a, b, i, j), inc, exc)


def test_debias_cls():
    from heracles.twopoint import debias_cls

    cls = {
        "a": np.zeros(100),
        "b": np.zeros(100),
        "c": np.zeros(
            (2, 100), dtype=np.dtype(float, metadata={"bias": 4.56, "spin_2": 2})
        ),
        "d": np.zeros(
            (2, 2, 3, 100), dtype=np.dtype(float, metadata={"spin_1": 2, "spin_2": 2})
        ),
        "e": np.zeros(
            (2, 2, 3, 100), dtype=np.dtype(float, metadata={"spin_1": 0, "spin_2": 0})
        ),
    }

    nbs = {
        "a": 1.23,
        "d": 7.89,
        "e": 7.89,
    }

    debias_cls(cls, nbs, inplace=True)

    np.testing.assert_array_equal(cls["a"], -1.23)

    np.testing.assert_array_equal(cls["b"], 0.0)

    np.testing.assert_array_equal(cls["c"][:, :2], 0.0)
    np.testing.assert_array_equal(cls["c"][:, 2:], -4.56)

    np.testing.assert_array_equal(cls["d"][0, 0, :, :2], 0.0)
    np.testing.assert_array_equal(cls["d"][0, 0, :, 2:], -7.89)
    np.testing.assert_array_equal(cls["d"][1, 1, :, :2], 0.0)
    np.testing.assert_array_equal(cls["d"][1, 1, :, 2:], -7.89)
    np.testing.assert_array_equal(cls["d"][0, 1, :, :], 0.0)
    np.testing.assert_array_equal(cls["d"][1, 0, :, :], 0.0)

    np.testing.assert_array_equal(cls["e"], -7.89)


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
        1: np.zeros((2, 100), dtype=np.dtype(float, metadata=md1)),
        2: np.zeros((2, 100), dtype=np.dtype(float, metadata=md2)),
        3: np.zeros((2, 100), dtype=np.dtype(float, metadata=md3)),
    }

    nbs = {
        1: 1.23,
        2: 4.56,
        3: 7.89,
    }

    debias_cls(cls, nbs, inplace=True)

    np.testing.assert_array_equal(cls[1][:, :2], 0.0)
    np.testing.assert_array_equal(cls[1][0, 2:], -1.23 / pw0[2:] / pw2[2:])
    np.testing.assert_array_equal(cls[1][1, 2:], -1.23 / pw0[2:] / pw2[2:])

    np.testing.assert_array_equal(cls[2][:, :2], 0.0)
    np.testing.assert_array_equal(cls[2][0, 2:], -4.56 / pw0[2:])
    np.testing.assert_array_equal(cls[2][1, 2:], -4.56 / pw0[2:])

    np.testing.assert_array_equal(cls[3][:, :2], 0.0)
    np.testing.assert_array_equal(cls[3][0, 2:], -7.89 / pw0[2:])
    np.testing.assert_array_equal(cls[3][1, 2:], -7.89 / pw0[2:])


@patch("convolvecl.mixmat_eb")
@patch("convolvecl.mixmat")
def test_mixing_matrices(mock, mock_eb, lmax, rng):
    from heracles.twopoint import mixing_matrices

    # this only tests the function logic
    # the mixing matrix computation itself is tested elsewhere

    # field definition, requires mask and spin weight

    # mixmat_eb returns three values
    mock.return_value = rng.random((lmax + 1, lmax + 1))
    mock_eb.return_value = rng.random((3, lmax + 1, lmax + 1))

    lmax = 20
    cl = rng.standard_normal(lmax + 1)

    # create the mock field information
    fields = {
        "POS": Mock(mask="VIS", spin=0),
        "SHE": Mock(mask="WHT", spin=2),
    }

    # compute pos-pos
    cls = {("VIS", "VIS", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 1
    assert mock.call_count == 1
    assert mock_eb.call_count == 0
    mock.assert_called_with(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0))
    assert mms["POS", "POS", 0, 1].array is mock.return_value
    assert mms["POS", "POS", 0, 1].axis == (0,)

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute pos-she
    cls = {("VIS", "WHT", 0, 1): cl, ("WHT", "VIS", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 2
    assert mock.call_count == 2
    assert mock_eb.call_count == 0
    assert mock.call_args_list == [
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 2)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(2, 0)),
    ]
    assert mms["POS", "SHE", 0, 1].array is mock.return_value
    assert mms["POS", "SHE", 0, 1].axis == (0,)
    assert mms["SHE", "POS", 0, 1].array is mock.return_value
    assert mms["SHE", "POS", 0, 1].axis == (0,)

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute she-she
    cls = {("WHT", "WHT", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 1
    assert mock.call_count == 0
    assert mock_eb.call_count == 1
    mock_eb.assert_called_with(cl, l1max=None, l2max=None, l3max=None, spin=(2, 2))
    assert mms["SHE", "SHE", 0, 1].array is mock_eb.return_value
    assert mms["SHE", "SHE", 0, 1].axis == (1,)

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute unknown
    cls = {("X", "Y", 0, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 0

    mock.reset_mock()
    mock_eb.reset_mock()

    # compute multiple combinations
    cls = {("VIS", "VIS", 0, 0): cl, ("VIS", "VIS", 0, 1): cl, ("VIS", "VIS", 1, 1): cl}
    mms = mixing_matrices(fields, cls)
    assert len(mms) == 3
    assert mock.call_count == 3
    assert mock_eb.call_count == 0
    assert mock.call_args_list == [
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
        call(cl, l1max=None, l2max=None, l3max=None, spin=(0, 0)),
    ]
    assert mms.keys() == {
        ("POS", "POS", 0, 0),
        ("POS", "POS", 0, 1),
        ("POS", "POS", 1, 1),
    }

    # test inversion of mixing matrices
    from heracles.twopoint import Result, invert_mixing_matrix

    cls = {("VIS", "VIS", 0, 0): cl, ("VIS", "VIS", 0, 1): cl, ("VIS", "VIS", 1, 1): cl}
    mms = mixing_matrices(fields, cls)
    for key in mms:
        _m = np.ones_like(mms[key].array)
        mms[key] = Result(_m, axis=mms[key].axis, ell=mms[key].ell)

    inv_mms = invert_mixing_matrix(mms)
    for key in mms:
        _inv_mms = np.sum(inv_mms[key].array)
        np.testing.assert_allclose(_inv_mms, 1.0)

    # test application of mixing matrices
    from heracles.twopoint import apply_mixing_matrix

    for key in cls:
        _cl = np.ones_like(cls[key])
        cls[key] = _cl

    mixed_cls = apply_mixing_matrix(cls, inv_mms)
    for key in cls:
        _mixed_cls = np.sum(mixed_cls[key].array)
        np.testing.assert_allclose(_mixed_cls, 1.0)
