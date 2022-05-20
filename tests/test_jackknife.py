import pytest


@pytest.fixture
def mock_writemask(mock_mask, datadir, nside):

    import fitsio

    filename = str(datadir / 'mask.fits')
    fits = fitsio.FITS(filename, 'rw')
    fits.write(data=None)
    fits.write_table([mock_mask], names=['WEIGHT'], header={'NSIDE': nside, 'ORDERING': 'RING', 'INDXSCHM': 'IMPLICIT', 'OBJECT': 'FULLSKY'})
    fits.close()

    return filename


def test_get_radec(mock_index, nside):

    from le3_pk_wl.jackknife import get_radec

    import numpy as np
    import healpy as hp

    mask_radec = get_radec(mock_index, nside)

    test_radec = np.transpose(hp.pixelfunc.pix2ang(nside, mock_index, lonlat=True))

    assert (mask_radec[:, 0] == test_radec[:, 0]).all()
    assert (mask_radec[:, 1] == test_radec[:, 1]).all()


def test_iterate_kmeans(mock_radec, options, jkparams):

    from le3_pk_wl.parameters import Params
    from le3_pk_wl.jackknife import iterate_kmeans

    params = Params(**options)

    ncen = 50

    km, ncent, ncut = iterate_kmeans(params, mock_radec)

    assert ncent <= ncen


def test_find_mask_regions(options, nside, jkparams, mock_radec, mock_mask):

    from le3_pk_wl.parameters import Params

    from le3_pk_wl.jackknife import iterate_kmeans, find_mask_regions

    params = Params(**options)

    km, ncent, ncut = iterate_kmeans(params, mock_radec)

    mask_labels = find_mask_regions(params, mock_mask, km.centers)
    assert (mask_labels <= ncent).all()


def test_kmeans_split(options, workdir, jkparams, mock_catalog, mock_writemask, zbins):

    from le3_pk_wl.parameters import Params

    from le3_pk_wl.jackknife import kmeans_split

    options['calc_pp'] = True
    options['calc_pe'] = True

    options['mask_name'] = mock_writemask

    params = Params(**options)

    kmeans_split(params)
