import pytest


NFIELDS_TEST = 4


@pytest.fixture(scope='session')
def mock_mask_fields(nside):
    import numpy as np
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.random.rand(npix*NFIELDS_TEST).reshape((npix, NFIELDS_TEST))
    pixels = np.unique(np.random.randint(0, npix, npix//3))
    maskpix = np.delete(np.arange(0, npix), pixels)
    for i in range(0, NFIELDS_TEST):
        maps[:, i][maskpix] = 0
    return [maps, pixels]


@pytest.fixture(scope='session')
def mock_writemask_partial(mock_mask_fields, nside, datadir):
    import fitsio

    filename = str(datadir / 'mask_partial.fits')

    maps, pixels = mock_mask_fields

    fits = fitsio.FITS(filename, 'rw')
    fits.write(data=None)
    fits.write_table([pixels, maps[:, 0][pixels], maps[:, 1][pixels], maps[:, 2][pixels], maps[:, 3][pixels]], names=['PIXEL', 'WEIGHT', 'FIELD1', 'FIELD2', 'FIELD3'], header={'NSIDE': nside, 'ORDERING': 'RING', 'INDXSCHM': 'EXPLICIT', 'OBJECT': 'PARTIAL'})
    fits.close()

    return filename


@pytest.fixture(scope='session')
def mock_writemask_full(mock_mask_fields, nside, datadir):
    import fitsio

    filename = str(datadir / 'mask_full.fits')

    maps, _ = mock_mask_fields

    fits = fitsio.FITS(filename, 'rw')
    fits.write(data=None)
    fits.write_table([maps[:, 0], maps[:, 1], maps[:, 2], maps[:, 3]], names=['WEIGHT', 'FIELD1', 'FIELD2', 'FIELD3'], header={'NSIDE': nside, 'ORDERING': 'RING', 'INDXSCHM': 'IMPLICIT', 'OBJECT': 'FULLSKY'})
    fits.close()

    return filename


@pytest.fixture(scope='session')
def mock_mask_extra(nside):
    import numpy as np
    import healpy as hp

    npix = hp.nside2npix(nside)
    maps = np.random.rand(npix)
    pixels = np.unique(np.random.randint(0, npix, npix//3))
    maskpix = np.delete(np.arange(0, npix), pixels)
    maps[maskpix] = 0
    return [maps, pixels]


@pytest.fixture(scope='session')
def mock_writemask_extra(mock_mask_extra, nside, datadir):
    import fitsio

    filename = str(datadir / 'mask_extra.fits')

    maps, _ = mock_mask_extra

    fits = fitsio.FITS(filename, 'rw')
    fits.write(data=None)
    fits.write_table([maps], names=['WEIGHT'], header={'NSIDE': nside, 'ORDERING': 'RING', 'INDXSCHM': 'IMPLICIT', 'OBJECT': 'FULLSKY'})
    fits.close()

    return filename


def test_write_read_binspec(tmp_path):

    from le3_pk_wl.io import write_binspec, read_binspec

    write_binspec('binspec.txt', 3, 12345, 'x == 5', workdir=str(tmp_path))
    bin_id, seed, query = read_binspec('binspec.txt', workdir=str(tmp_path))

    assert bin_id == 3
    assert seed == 12345
    assert query == 'x == 5'


def test_clobber_fits(tmp_path):

    import fitsio
    from le3_pk_wl.io import clobber_fits

    fits = fitsio.FITS(str(tmp_path / 'new.fits'), 'rw')
    fits.write(None)
    fits.write(None)
    assert len(fits) == 2
    fits.close()

    clobber_fits('new.fits', workdir=str(tmp_path))

    fits = fitsio.FITS(str(tmp_path / 'new.fits'), 'r')
    assert len(fits) == 1
    fits.close()


def test_write_read_header(tmp_path):

    import fitsio
    from le3_pk_wl.parameters import Params
    from le3_pk_wl.io import write_header, read_header

    # these are the options written to the header
    options = {
        'nside': 256,
        'lmin': 1,
        'lmax': 123,
        'nell_bins': 45,
        'linlogspace': 1,
        'nlsamp': 6,
        'nbar_cut': 0.7,
        'seed': 8910,
    }

    params = Params(**options)

    filename = 'test.fits'
    workdir = str(tmp_path)

    write_header(filename, params, workdir=workdir)

    assert (tmp_path / filename).exists()

    h = fitsio.read_header(str(tmp_path / filename))

    assert h['SOFTNAME'] == 'LE3_PK_WL'
    assert h['SOFTVERS'] == '1.0.0'
    assert h['NSIDE'] == 256
    assert h['LMIN'] == 1
    assert h['LMAX'] == 123
    assert h['NELLBIN'] == 45
    assert h['LOGLIN'] == 'log'
    assert h['NLSAMP'] == 6
    assert h['NBARCUT'] == 0.7
    assert h['SEED'] == 8910

    options_r = read_header(filename, workdir)

    assert options == options_r


def test_write_read_maps(tmp_path):

    import numpy as np
    import healpy as hp
    from le3_pk_wl.io import write_maps, read_maps

    nside = 4
    npix = 12*nside**2

    maps = {
        ('P', 1): np.random.rand(npix),
        ('V', 2): np.random.rand(npix),
        ('G1', 3): np.random.rand(npix),
    }

    write_maps('maps.fits', maps, workdir=str(tmp_path))
    assert (tmp_path / 'maps.fits').exists()
    maps_r = read_maps('maps.fits', workdir=str(tmp_path))

    assert maps.keys() == maps_r.keys()
    for key in maps:
        np.testing.assert_array_equal(maps[key], maps_r[key])

    # make sure map can be read by healpy
    m = hp.read_map(tmp_path / 'maps.fits', hdu='MAP0')
    np.testing.assert_array_equal(maps['P', 1], m)


def test_write_read_alms(mock_alms, tmp_path):

    import numpy as np
    from le3_pk_wl.io import write_alms, read_alms

    write_alms('alms.fits', mock_alms, workdir=str(tmp_path))
    assert (tmp_path / 'alms.fits').exists()
    alms = read_alms('alms.fits', workdir=str(tmp_path))

    assert alms.keys() == mock_alms.keys()
    for key in mock_alms:
        np.testing.assert_array_equal(mock_alms[key], alms[key])


def test_write_read_cls(mock_cls, tmp_path):

    from le3_pk_wl.io import write_cls, read_cls

    import numpy as np

    filename = 'test.fits'
    workdir = str(tmp_path)

    write_cls(filename, mock_cls, workdir=workdir)

    assert (tmp_path / filename).exists()

    cls = read_cls(filename, workdir=workdir)

    assert cls.keys() == mock_cls.keys()
    for key in mock_cls:
        np.testing.assert_array_equal(cls[key], mock_cls[key])


def test_write_read_mms(tmp_path):

    from le3_pk_wl.io import write_mms, read_mms

    import numpy as np

    filename = 'test.fits'
    workdir = str(tmp_path)

    mms = {
        ('00', 0, 1): np.random.randn(10, 10),
        ('0+', 1, 2): np.random.randn(20, 5),
        ('++', 2, 3): np.random.randn(10, 5, 2),
    }

    write_mms(filename, mms, workdir=workdir)

    assert (tmp_path / filename).exists()

    mms_ = read_mms(filename, workdir=workdir)

    assert mms_.keys() == mms.keys()
    for key in mms:
        np.testing.assert_array_equal(mms_[key], mms[key])


def test_write_read_cov(mock_cls, tmp_path):
    from itertools import combinations_with_replacement
    import numpy as np
    from le3_pk_wl.io import write_cov, read_cov

    workdir = str(tmp_path)

    cov = {}
    for k1, k2 in combinations_with_replacement(mock_cls, 2):
        cl1, cl2 = mock_cls[k1], mock_cls[k2]
        cov[k1, k2] = np.outer(cl1, cl2)

    filename = 'cov.fits'

    write_cov(filename, cov, workdir=workdir)

    assert (tmp_path / filename).exists()

    cov_ = read_cov(filename, workdir=workdir)

    assert cov_.keys() == cov.keys()
    for key in cov:
        np.testing.assert_array_equal(cov_[key], cov[key])


def test_read_mask_partial(mock_mask_fields, mock_writemask_partial, nside):

    from le3_pk_wl.io import read_mask
    import healpy as hp

    maps = mock_mask_fields[0]

    mask = read_mask(mock_writemask_partial, nside=nside)
    assert (mask == maps[:, 0]).all()

    ibin = 2
    mask = read_mask(mock_writemask_partial, nside=nside, field=ibin)
    assert (mask == maps[:, ibin]).all()

    ibin = 3
    mask = read_mask(mock_writemask_partial, nside=nside//2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside//2)
    assert (mask == maskud).all()


def test_read_mask_full(mock_mask_fields, mock_writemask_full, nside):

    from le3_pk_wl.io import read_mask
    import healpy as hp

    maps = mock_mask_fields[0]

    mask = read_mask(mock_writemask_full, nside=nside)
    assert (mask == maps[:, 0]).all()

    ibin = 2
    mask = read_mask(mock_writemask_full, nside=nside, field=ibin)
    assert (mask == maps[:, ibin]).all()

    ibin = 3
    mask = read_mask(mock_writemask_full, nside=nside//2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside//2)
    assert (mask == maskud).all()

    ibin = 3
    mask = read_mask(mock_writemask_full, nside=nside*2, field=ibin)
    maskud = hp.pixelfunc.ud_grade(maps[:, ibin], nside*2)
    assert (mask == maskud).all()


def test_read_mask_extra(mock_mask_fields, mock_mask_extra, mock_writemask_full, nside, mock_writemask_extra):

    from le3_pk_wl.io import read_mask

    maps = mock_mask_fields[0]
    maps_extra = mock_mask_extra[0]

    ibin = 2
    mask = read_mask(mock_writemask_full, nside=nside, field=ibin, extra_mask_name=mock_writemask_extra)
    assert (mask == maps[:, ibin]*maps_extra[:]).all()
