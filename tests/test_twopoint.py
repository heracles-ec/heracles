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


@pytest.fixture
def catalog(nside):

    size = 100_000
    ra = np.random.uniform(-180, 180, size=size)
    dec = np.degrees(np.arcsin(np.random.uniform(-1, 1, size=size)))
    g1 = 0.1*np.random.randn(size)
    g2 = 0.1*np.random.randn(size)
    w = np.random.uniform(0, 1, size=size)

    class MockCatalog:
        def __iter__(self):
            rows = unittest.mock.Mock()
            rows.size = size
            rows.ra = ra
            rows.dec = dec
            rows.g1 = g1
            rows.g2 = g2
            rows.w = w
            yield rows

    return MockCatalog()


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


def test_random_noisebias(catalog):

    from le3_pk_wl.twopoint import random_noisebias

    catalogs = {0: catalog}

    nside = 64

    nbs = random_noisebias('pg', nside, catalogs, repeat=5)

    assert len(nbs) == 6

    rows = next(iter(catalog))

    npix = 12*nside**2
    gvar = np.mean(rows.g1**2 + rows.g2**2)
    wtot = np.sum(rows.w**2)

    nb_pp = 4*np.pi/rows.size
    nb_ee = 2*np.pi/npix**2*gvar*wtot
    nb_bb = nb_ee

    np.testing.assert_allclose(nbs['PP', 0, 0], nb_pp, atol=0., rtol=0.05)
    np.testing.assert_allclose(nbs['PE', 0, 0], 0., atol=1e-6, rtol=0.)
    np.testing.assert_allclose(nbs['PB', 0, 0], 0., atol=1e-6, rtol=0.)
    np.testing.assert_allclose(nbs['EE', 0, 0], nb_ee, atol=0., rtol=0.05)
    np.testing.assert_allclose(nbs['BB', 0, 0], nb_bb, atol=0., rtol=0.05)
    np.testing.assert_allclose(nbs['EB', 0, 0], 0., atol=1e-7, rtol=0.)
