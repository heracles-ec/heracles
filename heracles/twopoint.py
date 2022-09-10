'''module for angular power spectrum estimation'''

import logging
import time
from datetime import timedelta
from itertools import product, combinations_with_replacement

import numpy as np
import healpy as hp
from scipy.stats import binned_statistic

from ._mixmat import mixmat, mixmat_eb
from .maps import update_metadata, map_catalogs as _map_catalogs, transform_maps as _transform_maps
from .util import toc_match

logger = logging.getLogger(__name__)


TWOPOINT_NAMES = list(map(''.join, combinations_with_replacement('PEBVW', 2)))
'''standard names for two-point functions (PE not EP etc.)'''


def angular_power_spectra(alms, alms2=None, *, lmax=None, include=None, exclude=None):
    '''compute angular power spectra from a set of alms'''

    logger.info('computing cls for %d%s alm(s)', len(alms), f'x{len(alms2)}' if alms2 is not None else '')
    t = time.monotonic()

    logger.info('using LMAX = %s for cls', lmax)

    # collect all alm combinations for computing cls
    if alms2 is None:
        alm_pairs = combinations_with_replacement(alms.items(), 2)
    else:
        alm_pairs = product(alms.items(), alms2.items())

    # compute cls for all alm pairs
    # do not compute duplicates
    cls = {}
    for ((n1, i1), alm1), ((n2, i2), alm2) in alm_pairs:
        # get the two-point code in standard order
        xy, yx = f'{n1}{n2}', f'{n2}{n1}'
        if xy not in TWOPOINT_NAMES and yx in TWOPOINT_NAMES:
            xy, yx = yx, xy
            i1, i2 = i2, i1
            n1, n2 = n2, n1

        # skip duplicate cls in any order
        if (xy, i1, i2) in cls or (yx, i2, i1) in cls:
            continue

        # check if cl is skipped by explicit include or exclude list
        if not toc_match((xy, i1, i2), include, exclude):
            continue

        logger.info('computing %s cl for bins %s, %s', xy, i1, i2)

        # compute the raw cl from the alms
        cl = hp.alm2cl(alm1, alm2, lmax_out=lmax)

        # collect metadata
        md = {}
        if alm1.dtype.metadata:
            for key, value in alm1.dtype.metadata.items():
                if key == 'noisbias':
                    md[key] = value if n1 == n2 and i1 == i2 else 0.
                else:
                    md[f'{key}_1'] = value
        if alm2.dtype.metadata:
            for key, value in alm2.dtype.metadata.items():
                if key == 'noisbias':
                    pass
                else:
                    md[f'{key}_2'] = value
        update_metadata(cl, **md)

        # add cl to the set
        cls[xy, i1, i2] = cl

    logger.info('computed %d cl(s) in %s', len(cls), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of cls
    return cls


def debias_cls(cls, noisebias=None, *, inplace=False):
    '''remove noise bias from cls'''

    logger.info('debiasing %d cl(s)%s', len(cls), ' in place' if inplace else '')
    t = time.monotonic()

    # toc dict for noise biases
    nbs = noisebias or {}

    # the output toc dict
    out = cls if inplace else {}

    # subtract noise bias of each cl in turn
    for key in cls:

        logger.info('debiasing %s cl for bins %s, %s', *key)

        cl = cls[key]
        md = cl.dtype.metadata or {}

        if not inplace:
            cl = cl.copy()
            update_metadata(cl, **md)

        # minimum l for correction
        lmin = max(abs(md.get('spin_1', 0)), abs(md.get('spin_2', 0)))

        # get noise bias from explicit dict, if given, or metadata
        nb = nbs.get(key, md.get('noisbias', 0.))

        # remove noise bias
        cl[lmin:] -= nb

        # write noise bias to corrected cl
        update_metadata(cl, noisbias=nb)

        # store debiased cl in output set
        out[key] = cl

    logger.info('debiased %d cl(s) in %s', len(out), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of debiased cls
    return out


def depixelate_cls(cls, *, inplace=False):
    '''remove discretisation kernel from cls'''

    logger.info('depixelate %d cl(s)%s', len(cls), ' in place' if inplace else '')
    t = time.monotonic()

    # keep a cache of convolution kernels (i.e. pixel window functions)
    fls = {
        'healpix': {},
    }

    # the output toc dict
    out = cls if inplace else {}

    # remove effect of convolution for each cl in turn
    for key in cls:

        logger.info('depixelate %s cl for bins %s, %s', *key)

        cl = cls[key]
        md = cl.dtype.metadata or {}

        if not inplace:
            cl = cl.copy()
            update_metadata(cl, **md)

        lmax = len(cl) - 1

        spins = [md.get('spin_1', 0), md.get('spin_2', 0)]
        kernels = [md.get('kernel_1'), md.get('kernel_2')]
        powers = [md.get('power_1', 0), md.get('power_2', 0)]
        areas = []

        # minimum l for corrections
        lmin = max(map(abs, spins))

        # deconvolve the kernels of the first and second map
        for i, spin, kernel in zip([1, 2], spins, kernels):
            logger.info('- spin-%s %s kernel', spin, kernel)
            if kernel is None:
                fl = None
                a = None
            elif kernel == 'healpix':
                nside = md[f'nside_{i}']
                if (nside, lmax, spin) not in fls[kernel]:
                    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)
                    fls[kernel][nside, lmax, 0] = fl0
                    fls[kernel][nside, lmax, 2] = fl2
                fl = fls[kernel].get((nside, lmax, spin))
                if fl is None:
                    logger.warning('no HEALPix kernel for NSIDE = %s, LMAX = %s, SPIN = %s', nside, lmax, spin)
                a = hp.nside2pixarea(nside)
            else:
                raise ValueError(f'unknown kernel: {kernel}')
            if fl is not None:
                cl[lmin:] /= fl[lmin:]
            areas.append(a)

        # scale by area**power
        for a, p in zip(areas, powers):
            if a is not None and p != 0:
                cl[lmin:] /= a**p

        # store depixelated cl in output set
        out[key] = cl

    logger.info('depixelated %d cl(s) in %s', len(out), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of depixelated cls
    return out


def mixing_matrices(cls, *, l1max=None, l2max=None, l3max=None):
    '''compute mixing matrices from a set of cls'''

    logger.info('computing two-point mixing matrices for %d cl(s)', len(cls))
    t = time.monotonic()

    logger.info('using L1MAX = %s, L2MAX = %s, L3MAX = %s', l1max, l2max, l3max)

    # set of computed mixing matrices
    mms = {}

    # go through the toc dict of cls and compute mixing matrices
    # which mixing matrix is computed depends on the combination of V/W maps
    for (xy, i1, i2), cl in cls.items():
        if xy == 'VV':
            logger.info('computing 00 mixing matrix for bins %s, %s', i1, i2)
            w00 = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max)
            mms['00', i1, i2] = w00
        elif xy == 'VW':
            logger.info('computing 0+ mixing matrix for bins %s, %s', i1, i2)
            w0p = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max, spin=(0, 2))
            mms['0+', i1, i2] = w0p
        elif xy == 'WV':
            logger.info('computing 0+ mixing matrix for bins %s, %s', i2, i1)
            w0p = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max, spin=(2, 0))
            mms['0+', i2, i1] = w0p
        elif xy == 'WW':
            logger.info('computing ++, --, +- mixing matrices for bins %s, %s', i1, i2)
            wpp, wmm, wpm = mixmat_eb(cl, l1max=l1max, l2max=l2max, l3max=l3max, spin=(2, 2))
            mms['++', i1, i2] = wpp
            mms['--', i1, i2] = wmm
            mms['+-', i1, i2] = wpm
        else:
            logger.warning('computing unknown %s mixing matrix for bins %s, %s', xy, i1, i2)
            w = mixmat(cl, l1max=l1max, l2max=l2max, l3max=l3max)
            mms[xy, i1, i2] = w

    logger.info('computed %d mm(s) in %s', len(mms), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of mixing matrices
    return mms


def pixelate_mms_healpix(mms, nside, *, inplace=False):
    '''apply HEALPix pixel window function to mms'''

    logger.info('pixelate %d mm(s)%s', len(mms), ' in place' if inplace else '')
    logger.info('kernel: HEALPix, NSIDE=%d', nside)
    t = time.monotonic()

    # pixel window functions
    lmax = 4*nside
    fl0, fl2 = hp.pixwin(nside, lmax=lmax, pol=True)

    # will be multiplied over rows
    fl0 = fl0[:, np.newaxis]
    fl2 = fl2[:, np.newaxis]

    # the output toc dict
    out = mms if inplace else {}

    # apply discretisation kernel from cl to each mm in turn
    for key in mms:

        logger.info('pixelate %s mm for bins %s, %s', *key)

        mm = mms[key]
        if not inplace:
            mm = mm.copy()

        n = np.shape(mm)[-2]
        if n >= lmax:
            logger.error('no HEALPix pixel window function for NSIDE=%d and LMAX=%d', nside, n-1)

        name = key[0]
        if name == '00':
            mm *= fl0[:n]*fl0[:n]
        elif name in ['0+', '+0']:
            mm *= fl0[:n]*fl2[:n]
        elif name in ['++', '--', '+-', '-+']:
            mm *= fl2[:n]*fl2[:n]
        else:
            logger.warning('unknown mixing matrix, assuming spin-0')
            mm *= fl0[:n]*fl0[:n]

        # store pixelated mm in output set
        out[key] = mm

    logger.info('pixelated %d mm(s) in %s', len(out), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of modified cls
    return out


def binned_cl(cl, bins, cmblike=False):
    '''compute a binned angular power spectrum'''

    ell = np.arange(len(cl))

    # cmblike bins l(l + 1) C_l / 2pi
    if cmblike:
        cl = ell*(ell+1)*cl/(2*np.pi)

    return binned_statistic(ell, cl, bins=bins, statistic='mean')[0]


def random_noisebias(which, nside, catalogs, vmaps=None, *, lmax=None, repeat=1,
                     overdensity=True, full=False):
    '''simple noise bias estimate from randomised position and shear maps'''

    if not all(k.upper() in ['P', 'G'] for k in which):
        raise ValueError('can only estimate noise bias for position (P) and shear (G) maps')

    if lmax is None:
        lmax = nside

    logger.info('estimating two-point noise bias for %d catalog(s)', len(catalogs))
    logger.info('randomising %s maps', ', '.join(map(str.upper, which)))
    logger.info('using NSIDE = %s', nside)
    logger.info('given %s visibility map(s)', 'no' if vmaps is None else len(vmaps))
    t = time.monotonic()

    if full:
        logger.info('estimating cross-noise biases')
        include = None
    else:
        include = [('PP', ..., ...), ('EE', ..., ...), ('BB', ..., ...)]

    nbs = {}

    for n in range(repeat):

        logger.info('estimating noise bias from randomised maps%s', '' if n == 0 else f' (repeat {n})')

        maps = _map_catalogs(which, nside, catalogs, vmaps, overdensity=overdensity, random=True)
        alms = _transform_maps(maps, lmax=lmax)
        cls = angular_power_spectra(alms, lmax=lmax, include=include)

        for k, cl in cls.items():
            ell = np.arange(2, cl.shape[-1])
            nb = np.sum((2*ell+1)*cl[2:])/np.sum(2*ell+1)
            nbs[k] = nbs.get(k, 0.) + (nb - nbs.get(k, 0.))/(n + 1)

    logger.info('estimated %d two-point noise biases in %s', len(nbs), timedelta(seconds=(time.monotonic() - t)))

    return nbs
