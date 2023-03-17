import unittest.mock
import pytest
import numpy as np


@pytest.fixture
def nside():
    return 32


@pytest.fixture
def zbins():
    zbins = {0: (0., 0.8), 1: (1.0, 1.2)}
    return zbins


@pytest.fixture
def mock_alms(zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    names = ['P', 'E', 'B']

    alms = {}
    for n in names:
        for i in zbins:
            a = np.random.randn(Nlm, 2) @ [1, 1j]
            a.dtype = np.dtype(a.dtype, metadata={'nside': 32})
            alms[n, i] = a

    return alms


def test_angular_power_spectra(mock_alms):
    from itertools import combinations_with_replacement
    from le3_pk_wl.twopoint import angular_power_spectra

    # alms cross themselves

    comb = set((f'{n1}{n2}', i1, i2) for (n1, i1), (n2, i2) in combinations_with_replacement(mock_alms, 2))

    cls = angular_power_spectra(mock_alms)

    assert cls.keys() == comb

    # explicit cross

    cls = angular_power_spectra(mock_alms, mock_alms)

    assert cls.keys() == comb

    # explicit include

    cls = angular_power_spectra(mock_alms, include=[('PP', ..., ...), ('PE', ..., ...)])

    assert cls.keys() == {(n, i1, i2) for n, i1, i2 in comb if n in ['PP', 'PE']}

    cls = angular_power_spectra(mock_alms, include=[('PP', 0), ('PE', 1)])

    assert cls.keys() == {(n, i1, i2) for n, i1, i2 in comb if (n, i1) in [('PP', 0), ('PE', 1)]}

    # explicit exclude

    cls = angular_power_spectra(mock_alms, exclude=[('PP',), ('PE',), ('PB',)])

    assert cls.keys() == {(n, i1, i2) for n, i1, i2 in comb if n not in ['PP', 'PE', 'PB']}

    cls = angular_power_spectra(mock_alms, exclude=[(..., 1, ...)])

    assert cls.keys() == {(n, i1, i2) for n, i1, i2 in comb if i1 != 1}


def test_debias_cls():

    from le3_pk_wl.twopoint import debias_cls

    cls = {
        ('PP', 0, 0): np.zeros(100),
    }

    nbs = {
        ('PP', 0, 0): 1.23,
    }

    debias_cls(cls, nbs, inplace=True)

    assert np.all(cls['PP', 0, 0] == -1.23)


def test_mixing_matrices():

    from le3_pk_wl.twopoint import mixing_matrices

    # this only tests the function logic
    # the mixing matrix computation itself is tested elsewhere

    lmax = 20
    cl = np.random.randn(lmax+1)

    # compute pos-pos
    cls = {('VV', 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 1
    assert mms['00', 0, 1].shape == (lmax+1, lmax+1)

    # compute pos-she
    cls = {('VW', 0, 1): cl, ('WV', 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 2
    assert mms['0+', 0, 1].shape == (lmax+1, lmax+1)
    assert mms['0+', 1, 0].shape == (lmax+1, lmax+1)

    # compute she-she
    cls = {('WW', 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 3
    assert mms['++', 0, 1].shape == (lmax+1, lmax+1)
    assert mms['--', 0, 1].shape == (lmax+1, lmax+1)
    assert mms['+-', 0, 1].shape == (lmax+1, lmax+1)
    np.testing.assert_allclose(mms['+-', 0, 1], mms['++', 0, 1] + mms['--', 0, 1])

    # compute unknown
    cls = {('XY', 0, 1): cl}
    mms = mixing_matrices(cls)
    assert len(mms) == 1
    assert mms['XY', 0, 1].shape == (lmax+1, lmax+1)


def test_pixelate_mms_healpix():

    import healpy as hp
    from le3_pk_wl.twopoint import pixelate_mms_healpix

    nside = 512
    lmax = 1000

    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)

    mms = {
        ('00', 0, 0): np.eye(lmax+1),
        ('0+', 0, 0): np.eye(lmax+1),
        ('++', 0, 0): np.eye(lmax+1),
        ('--', 0, 0): np.eye(lmax+1),
        ('+-', 0, 0): np.eye(lmax+1),
        ('ab', 0, 0): np.eye(lmax+1),
    }

    pixelate_mms_healpix(mms, nside, inplace=True)

    assert np.all(mms['00', 0, 0] == np.diag(fl0*fl0))
    assert np.all(mms['0+', 0, 0] == np.diag(fl0*fl2))
    assert np.all(mms['++', 0, 0] == np.diag(fl2*fl2))
    assert np.all(mms['--', 0, 0] == np.diag(fl2*fl2))
    assert np.all(mms['+-', 0, 0] == np.diag(fl2*fl2))
    assert np.all(mms['ab', 0, 0] == np.diag(fl0*fl0))


@pytest.mark.parametrize('cmblike', [False, True])
def test_binned_cl(cmblike):

    from le3_pk_wl.twopoint import binned_cl

    cl = np.random.randn(21)

    bins = [0, 5, 10, 15, 20]

    result = binned_cl(cl, bins, cmblike)

    ell = np.arange(len(cl))

    if cmblike:
        cl *= ell*(ell+1)/(2*np.pi)

    begin = bins[:-1]
    end = bins[1:]
    end[-1] = np.nextafter(end[-1], end[-1]+1)

    binned = []
    for a, b in zip(begin, end):
        binned.append(cl[(a <= ell) & (ell < b)].mean())

    np.testing.assert_array_almost_equal(result, binned)


@pytest.mark.parametrize('full', [False, True])
def test_random_noisebias(full):

    from le3_pk_wl.twopoint import random_noisebias

    nside = 64
    npix = 12*nside**2

    catalog = unittest.mock.Mock()
    catalog.visibility = None

    map_a = unittest.mock.Mock(side_effect=lambda _: np.random.rand(npix))
    map_b = unittest.mock.Mock(side_effect=lambda _: np.random.rand(npix))

    initial_randomize = [map_a.randomize, map_b.randomize]

    maps = {'A': map_a, 'B': map_b}
    catalogs = {0: catalog, 1: catalog}

    nbs = random_noisebias(maps, catalogs, repeat=5, full=full)

    for m, r in zip(maps.values(), initial_randomize):
        assert m.randomize is r

    keys = [('AA', 0, 0), ('AA', 0, 1), ('AA', 1, 1),
            ('BB', 0, 0), ('BB', 0, 1), ('BB', 1, 1)]
    if full:
        keys += [('AB', 0, 0), ('AB', 1, 1), ('AB', 0, 1), ('AB', 1, 0)]

    assert set(nbs.keys()) == set(keys)

    # filter with include and exclude

    nbs = random_noisebias(maps, catalogs, repeat=5, full=full,
                           include=[('A', 0), ('B', ...)], exclude=[('B', 0)])

    keys = [('AA', 0, 0), ('BB', 1, 1)]
    if full:
        keys += [('AB', 0, 1)]

    assert set(nbs.keys()) == set(keys)
