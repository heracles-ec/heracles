import pytest
from numba import config

config.DISABLE_JIT = True


@pytest.fixture(scope='session')
def options_all():
    return {}


@pytest.fixture(scope='session')
def nside(options_all):
    nside = 32
    options_all['nside'] = nside
    return nside


@pytest.fixture(scope='session')
def datadir(tmp_path_factory):
    datadir = tmp_path_factory.mktemp('data')
    return datadir


@pytest.fixture
def options(options_all):
    return options_all.copy()


@pytest.fixture
def zbins(options):
    zbins = {0: (0., 0.8), 1: (1.0, 1.2)}
    options['zselect'] = zbins
    return zbins


@pytest.fixture
def jkparams(options):

    lmin = 2

    options['lmin'] = lmin
    options['nsideKmeans'] = 16
    options['nClustersMax'] = 50
    options['nClustersMin'] = 22


@pytest.fixture
def workdir(tmp_path, options):
    options['workdir'] = str(tmp_path)
    return tmp_path


@pytest.fixture(scope='session')
def mock_mask(nside):
    import numpy as np
    import healpy as hp
    return np.random.rand(hp.nside2npix(nside))


@pytest.fixture(scope='session')
def mock_catalog_all(datadir):
    import numpy as np
    import fitsio

    filename = str(datadir / 'catalog.fits')

    ngal = 100000
    tx = np.random.rand(ngal)*360.0
    ty = -90.0 + np.random.rand(ngal)*180.0
    tz = 0.6 + np.random.rand(ngal)*0.6
    tg1 = 3e-3 * (-1 + np.random.rand(ngal)*2)
    tg2 = 3e-3 * (-1 + np.random.rand(ngal)*2)
    tw = np.random.rand(ngal)

    fits = fitsio.FITS(filename, 'rw')
    fits.write([tx, ty, tg1, tg2, tz, tw], names=['RIGHT_ASCENSION', 'DECLINATION', 'G1', 'G2', 'z', 'WEIGHT'])
    fits.close()

    return filename


@pytest.fixture
def mock_catalog(options, mock_catalog_all):
    options['cat_name'] = mock_catalog_all
    options['cat_name_pos'] = mock_catalog_all
    return mock_catalog_all


@pytest.fixture(scope='session')
def mock_catalog_nans(datadir):
    import numpy as np
    import fitsio

    filename = str(datadir / 'catalog_nans.fits')

    ngal = 100000
    tx = np.random.rand(ngal)*360.0
    ty = -90.0 + np.random.rand(ngal)*180.0
    tz = 0.6 + np.random.rand(ngal)*0.6
    tg1 = 3e-3 * (-1 + np.random.rand(ngal)*2)
    tg2 = 3e-3 * (-1 + np.random.rand(ngal)*2)
    tw = np.random.rand(ngal)

    tx[np.random.randint(ngal, size=50)] = np.nan
    ty[np.random.randint(ngal, size=50)] = np.nan

    fits = fitsio.FITS(filename, 'rw')
    fits.write([tx, ty, tg1, tg2, tz, tw], names=['RIGHT_ASCENSION', 'DECLINATION', 'G1', 'G2', 'z', 'WEIGHT'])
    fits.close()

    return filename


@pytest.fixture
def mock_catalog_invalid_rows(options, mock_catalog_nans):
    options['cat_name'] = mock_catalog_nans
    options['cat_name_pos'] = mock_catalog_nans
    return mock_catalog_nans


@pytest.fixture
def mock_alms(zbins):
    import numpy as np

    lmax = 32

    Nlm = (lmax + 1) * (lmax + 2) // 2

    names = ['P', 'E', 'B']

    alms = {}
    for n in names:
        for i in zbins:
            alms[n, i] = np.random.randn(Nlm, 2) @ [1, 1j]

    return alms


@pytest.fixture
def mock_cls(options, mock_alms):
    from itertools import combinations_with_replacement
    import healpy as hp

    cls = {}
    for (n1, i1), (n2, i2) in combinations_with_replacement(mock_alms, 2):
        cls[f'{n1}{n2}', i1, i2] = hp.alm2cl(mock_alms[n1, i1], mock_alms[n2, i2])
    return cls


@pytest.fixture(scope='session')
def mock_index(nside):
    import numpy as np
    import healpy as hp

    index = np.unique(np.random.randint(0, hp.pixelfunc.nside2npix(nside), 1000))

    return index


@pytest.fixture(scope='session')
def mock_radec():

    import numpy as np

    npix = 10000
    tx = -180.0 + np.random.rand(npix)*360.0
    ty = -90.0 + np.random.rand(npix)*180.0

    mock_radec = np.vstack((tx, ty)).T

    return mock_radec


@pytest.fixture(scope='session')
def mock_labels():
    import numpy as np

    labels = np.random.randint(0, 50, 10000)

    return labels


@pytest.fixture(scope='session')
def mock_centers():
    import numpy as np
    centers = np.random.rand(50, 2)
    return centers


@pytest.fixture
def mock_alms_ps(options, workdir):

    import numpy as np
    import healpy as hp

    lmax = 32
    options['l3max'] = lmax

    Nlm = (lmax + 1) * (lmax + 2) // 2
    alms = np.random.rand(4, Nlm) + np.random.rand(4, Nlm)*1j

    hp.write_alm(str(workdir / 'alms_p1.fits'), alms[0, :])
    hp.write_alm(str(workdir / 'alms_p2.fits'), alms[1, :])
    hp.write_alm(str(workdir / 'alms_s1.fits'), alms[2, :])
    hp.write_alm(str(workdir / 'alms_s2.fits'), alms[3, :])

    return ['alms_p1.fits', 'alms_p2.fits', 'alms_s1.fits', 'alms_s2.fits']
