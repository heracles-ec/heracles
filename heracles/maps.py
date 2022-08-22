'''module for map-making'''

import time
from datetime import timedelta
import logging
import numpy as np
import healpy as hp
from numba import njit, int64, float64


logger = logging.getLogger(__name__)


@njit((float64[:], int64[:]), nogil=True)
def _map_pos(m, ipix):
    for i in ipix:
        m[i] += 1


@njit((float64[:], float64[:, :], int64[:], float64[:], float64[:], float64[:]), nogil=True)
def _map_she(n, m, ipix, w, g1, g2):
    for i, w_i, g1_i, g2_i in zip(ipix, w, g1, g2):
        n[i] += w_i
        m[0, i] += w_i/n[i]*(g1_i - m[0, i])
        m[1, i] += w_i/n[i]*(g2_i - m[1, i])


@njit((float64[:], int64[:], float64[:]), nogil=True)
def _map_wht(m, ipix, w):
    for i, w_i in zip(ipix, w):
        m[i] += w_i


def visibility_map(nside, vmap):
    '''create a visibility map with metadata for analysis'''

    logger.info('copying visibility map')
    t = time.monotonic()

    nside_in = hp.get_nside(vmap)
    if nside != nside_in:
        logger.info('changing NSIDE of visibility map from %s to %s', nside_in, nside)
        vmap = hp.ud_grade(vmap, nside)
    else:
        vmap = np.copy(vmap)

    logger.info('visibility map copied in %s', timedelta(seconds=(time.monotonic() - t)))

    update_metadata(vmap, spin=0, kernel='healpix', power=0)

    return vmap


def map_positions(nside, catalog, vmap=None, *, random=False, overdensity=True):
    '''map positions from catalog to HEALPix map'''

    logger.info('mapping positions')
    t = time.monotonic()

    # number of pixels for nside
    npix = hp.nside2npix(nside)

    # keep track of the total number of galaxies
    ngal = 0

    # map positions
    pos = np.zeros(npix, dtype=np.float64)
    for rows in catalog:
        if not random:
            ipix = hp.ang2pix(nside, rows.ra, rows.dec, lonlat=True)
            _map_pos(pos, ipix)
        ngal += rows.size

    # randomise position map if asked to
    if random:
        logger.info('randomising positions')
        if vmap is None:
            p = np.full(npix, 1/npix)
        else:
            p = vmap/np.sum(vmap)
        pos[:] = np.random.multinomial(ngal, p)

    # compute average number density
    nbar = ngal/npix
    if vmap is not None:
        nbar /= np.mean(vmap)

    # compute overdensity if asked to
    if overdensity:
        logger.info('computing overdensity maps')
        pos /= nbar
        if vmap is None:
            pos -= 1
        else:
            pos -= vmap
        power = 0
    else:
        power = 1

    logger.info('mapped %s positions in %s', f'{ngal:_}', timedelta(seconds=(time.monotonic() - t)))

    # set metadata of array
    update_metadata(pos, spin=0, nbar=nbar, kernel='healpix', power=power)

    # return the position map
    return pos


def map_shears(nside, catalog, *, random=False, normalize=True):
    '''map shears from catalog to HEALPix maps'''

    logger.info('mapping shears')
    t = time.monotonic()

    if random:
        logger.info('randomising shears')

    # number of pixels for nside
    npix = hp.nside2npix(nside)

    # keep track of the total number of galaxies
    ngal = 0

    # create the weight and shear map
    wht = np.zeros(npix, dtype=np.float64)
    she = np.zeros((2, npix), dtype=np.float64)

    # go through rows in catalogue and get the shear columns
    # randomise if asked to
    # do the mapping
    for rows in catalog:
        if rows.g1 is None:
            raise TypeError('map_shears: catalogue does not contain shears')

        g1 = np.asanyarray(rows.g1, dtype=np.float64)
        g2 = np.asanyarray(rows.g2, dtype=np.float64)

        if rows.w is None:
            w = np.ones(rows.size, dtype=np.float64)
        else:
            w = np.asanyarray(rows.w, dtype=np.float64)

        if random:
            a = np.random.uniform(0., 2*np.pi, size=rows.size)
            r = np.hypot(g1, g2)
            g1, g2 = r*np.cos(a), r*np.sin(a)
            del a, r

        ipix = hp.ang2pix(nside, rows.ra, rows.dec, lonlat=True)
        _map_she(wht, she, ipix, w, g1, g2)
        ngal += rows.size

    # compute average weight in nonzero pixels
    wbar = wht.mean()

    # normalise the weight in each pixel if asked to
    if normalize:
        wht /= wbar
        power = 0
    else:
        power = 1

    # shear was averaged in each pixel for numerical stability
    # now compute the sum
    she *= wht

    # set metadata of array
    update_metadata(she, spin=2, wbar=wbar, kernel='healpix', power=power)

    logger.info('mapped %s shears in %s', f'{ngal:_}', timedelta(seconds=(time.monotonic() - t)))

    # return the shear map
    return she


def map_weights(nside, catalog, normalize=True):
    '''map weights from catalog to HEALPix map'''

    logger.info('mapping weights')
    t = time.monotonic()

    # number of pixels for nside
    npix = hp.nside2npix(nside)

    # create the weight map
    wht = np.zeros(npix, dtype=np.float64)
    for rows in catalog:
        if rows.w is None:
            w = np.ones(rows.size, dtype=np.float64)
        else:
            w = np.asanyarray(rows.w, dtype=np.float64)
        ipix = hp.ang2pix(nside, rows.ra, rows.dec, lonlat=True)
        _map_wht(wht, ipix, w)

    # compute average weight in nonzero pixels
    wbar = wht.mean()

    # normalise the weight in each pixel if asked to
    if normalize:
        wht /= wbar
        power = 0
    else:
        power = 1

    # set metadata of arrays
    update_metadata(wht, spin=0, wbar=wbar, kernel='healpix', power=power)

    logger.info('weights mapped in %s', timedelta(seconds=(time.monotonic() - t)))

    # return the weight map
    return wht


def transform_maps(maps, **kwargs):
    '''transform a set of maps to alms'''

    logger.info('transforming %d map(s) to alms', len(maps))
    t = time.monotonic()

    # convert maps to alms, taking care of complex and spin-weighted maps
    alms = {}
    for (n, i), m in maps.items():

        nside = hp.get_nside(m)

        md = m.dtype.metadata or {}
        spin = md.get('spin', 0)

        logger.info('transforming %s map (spin %s) for bin %s', n, spin, i)

        if spin == 0:
            pol = False
        elif spin == 2:
            pol = True
            m = [np.zeros(np.shape(m)[-1]), m[0], m[1]]
        else:
            raise NotImplementedError(f'spin-{spin} maps not yet supported')

        a = hp.map2alm(m, pol=pol, **kwargs)

        if spin == 0:
            alms[n, i] = a
            if md:
                update_metadata(alms[n, i], nside=nside, **md)
        elif spin == 2:
            alms['E', i], alms['B', i] = a[1], a[2]
            if md:
                update_metadata(alms['E', i], nside=nside, **md)
                update_metadata(alms['B', i], nside=nside, **md)

    logger.info('transformed %d map(s) in %s', len(alms), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of alms
    return alms


def update_metadata(array, **metadata):
    '''update metadata of an array dtype'''
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    md.update(metadata)
    array.dtype = np.dtype(array.dtype, metadata=md)


def map_catalogs(which, nside, catalogs, vmaps=None, *, overdensity=True, random=False):
    '''map a set of catalogues

    The `which` argument is a string that selects maps to be produced: `P` for
    positions, `G` for shears, `V` for visibilities, and `W` for weights.

    '''

    logger.info('mapping %d catalog(s)', len(catalogs))
    logger.info('creating %s map(s)', ', '.join(map(str.upper, which)))
    logger.info('using NSIDE = %s', nside)
    logger.info('given %s visibility map(s)', 'no' if vmaps is None else len(vmaps))
    t = time.monotonic()

    # the toc dict of maps
    maps = {}

    # for output, set up the toc dict in map-first order
    for k in map(str.upper, which):
        for i in catalogs:
            maps[k, i] = None

    # for computation, go through catalogues first
    for i, c in catalogs.items():

        logger.info('mapping catalog for bin %s', i)

        # try and find the visibility map for this bin
        if vmaps is not None:
            if i in vmaps:
                v = vmaps[i]
            elif None in vmaps:
                v = vmaps[None]
            else:
                v = None
        else:
            v = None

        if v is not None:
            logger.info('found visibility map for bin %s', i)
        else:
            logger.info('no visibility map for bin %s', i)

        # compute the individual maps
        for k in map(str.upper, which):
            if k == 'P':
                m = map_positions(nside, c, v, overdensity=overdensity, random=random)
            elif k == 'G':
                m = map_shears(nside, c, random=random)
            elif k == 'V':
                if v is None:
                    raise KeyError(f'missing visibility map for catalog {i}')
                m = visibility_map(nside, v)
            elif k == 'W':
                m = map_weights(nside, c)
            else:
                raise ValueError(f'unknown map code: {k}')
            maps[k, i] = m

    logger.info('created %d map(s) in %s', len(maps), timedelta(seconds=(time.monotonic() - t)))

    # return maps as a toc dict
    return maps
