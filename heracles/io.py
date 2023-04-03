'''module for file reading and writing'''

import os
import logging

import numpy as np
import healpy as hp
import fitsio

from .util import toc_match

logger = logging.getLogger(__name__)


_METADATA_COMMENTS = {
    'spin': 'spin weight of map',
    'kernel': 'mapping kernel of map',
    'power': 'area power of map',
    'spin_1': 'spin weight of first map',
    'kernel_1': 'mapping kernel of first map',
    'power_1': 'area power of first map',
    'spin_2': 'spin weight of second map',
    'kernel_2': 'mapping kernel of second map',
    'power_2': 'area power of second map',
    'noisbias': 'noise bias of spectrum',
}


def _write_metadata(hdu, metadata):
    '''write array metadata to FITS HDU'''
    md = metadata or {}
    for key, value in md.items():
        hdu.write_key('META ' + key.upper(), value, _METADATA_COMMENTS.get(key))


def _read_metadata(hdu):
    '''read array metadata from FITS HDU'''
    h = hdu.read_header()
    md = {}
    for key in h:
        if key.startswith('META '):
            md[key[5:].lower()] = h[key]
    return md


def read_mask(mask_name, nside=None, field=0, extra_mask_name=None):
    '''read visibility map from a HEALPix map file'''
    mask = hp.read_map(mask_name, field=field)

    # set unseen pixels to zero
    unseen = np.where(mask == hp.UNSEEN)
    mask[unseen] = 0

    nside_mask = hp.get_nside(mask)

    if nside is not None:
        # mask is provided at a different resolution
        if nside_mask < nside:
            print('WARNING: Nside of mask < Nside of requested maps')
        if nside_mask != nside:
            mask = hp.ud_grade(mask, nside)
            nside_mask = nside

    # apply extra mask if given
    if extra_mask_name is not None:
        extra_mask = hp.read_map(extra_mask_name)
        nside_extra = hp.get_nside(extra_mask)
        if nside_extra != nside_mask:
            extra_mask = hp.ud_grade(extra_mask, nside_mask)
        mask *= extra_mask

    return mask


def write_maps(filename, maps, *, clobber=False, workdir='.', include=None, exclude=None):
    '''write a set of maps to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    '''

    logger.info('writing %d maps to %s', len(maps), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode='rw', clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode='rw', clobber=False) as fits:

        # write a new TOC extension if FITS doesn't already contain one
        if 'MAPTOC' not in fits:
            fits.create_table_hdu(names=['EXT', 'NAME', 'BIN'],
                                  formats=['10A', '10A', 'I'],
                                  extname='MAPTOC')

        # get a recarray to write TOC entries with
        tocentry = np.empty(1, dtype=fits['MAPTOC'].get_rec_dtype()[0])

        # get the first free map extension index
        mapn = 0
        while f'MAP{mapn}' in fits:
            mapn += 1

        # write every map
        for (n, i), m in maps.items():

            # skip if not selected
            if not toc_match((n, i), include=include, exclude=exclude):
                continue

            logger.info('writing %s map for bin %s', n, i)

            # the cl extension name
            ext = f'MAP{mapn}'
            mapn += 1

            # prepare column data and names
            cols = list(np.atleast_2d(m))
            if len(cols) == 1:
                colnames = [n]
            else:
                colnames = [f'{n}{j+1}' for j in range(len(cols))]

            # write the data
            fits.write_table(cols, names=colnames, extname=ext)

            # write the metadata
            _write_metadata(fits[ext], m.dtype.metadata)

            # HEALPix metadata
            npix = np.shape(m)[-1]
            nside = hp.npix2nside(npix)
            fits[ext].write_key('PIXTYPE', 'HEALPIX', 'HEALPIX pixelisation')
            fits[ext].write_key('ORDERING', 'RING', 'Pixel ordering scheme, either RING or NESTED')
            fits[ext].write_key('NSIDE', nside, 'Resolution parameter of HEALPIX')
            fits[ext].write_key('FIRSTPIX', 0, 'First pixel # (0 based)')
            fits[ext].write_key('LASTPIX', npix-1, 'Last pixel # (0 based)')
            fits[ext].write_key('INDXSCHM', 'IMPLICIT', 'Indexing: IMPLICIT or EXPLICIT')
            fits[ext].write_key('OBJECT', 'FULLSKY', 'Sky coverage, either FULLSKY or PARTIAL')

            # write the TOC entry
            tocentry[0] = (ext, n, i)
            fits['MAPTOC'].append(tocentry)

    logger.info('done with %d maps', len(maps))


def read_maps(filename, workdir='.', *, include=None, exclude=None):
    '''read a set of maps from a FITS file'''

    logger.info('reading maps from %s', filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of maps
    maps = {}

    # open the FITS file for reading
    with fitsio.FITS(path) as fits:

        # get the TOC from the FITS file
        fits_toc = fits['MAPTOC'].read()

        # read every entry in the TOC, add it to the list, then read the maps
        for entry in fits_toc:
            ext, n, i = entry[['EXT', 'NAME', 'BIN']]

            # skip if not selected
            if not toc_match((n, i), include=include, exclude=exclude):
                continue

            logger.info('reading %s map for bin %s', n, i)

            # read the map from the extension
            m = fits[ext].read()

            # turn the structured array of columns into an unstructured array
            # transpose so that columns become rows (as that is how maps are)
            # then squeeze out degenerate axes
            m = np.squeeze(np.lib.recfunctions.structured_to_unstructured(m).T)

            # read and attach metadata
            m.dtype = np.dtype(m.dtype, metadata=_read_metadata(fits[ext]))

            # store in set of maps
            maps[n, i] = m

    logger.info('done with %d maps', len(maps))

    # return the dictionary of maps
    return maps


def write_alms(filename, alms, *, clobber=False, workdir='.', include=None, exclude=None):
    '''write a set of alms to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    '''

    logger.info('writing %d alms to %s', len(alms), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode='rw', clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode='rw', clobber=False) as fits:

        # write a new TOC extension if FITS doesn't already contain one
        if 'ALMTOC' not in fits:
            fits.create_table_hdu(names=['EXT', 'NAME', 'BIN'],
                                  formats=['10A', '10A', 'I'],
                                  extname='ALMTOC')

        # get a recarray to write TOC entries with
        tocentry = np.empty(1, dtype=fits['ALMTOC'].get_rec_dtype()[0])

        # get the first free alm extension index
        almn = 0
        while f'ALM{almn}' in fits:
            almn += 1

        # write every alm
        for (n, i), alm in alms.items():

            # skip if not selected
            if not toc_match((n, i), include=include, exclude=exclude):
                continue

            logger.info('writing %s alm for bin %s', n, i)

            # the cl extension name
            ext = f'ALM{almn}'
            almn += 1

            # write the data
            fits.write_table([alm.real, alm.imag], names=['real', 'imag'], extname=ext)

            # write the metadata
            _write_metadata(fits[ext], alm.dtype.metadata)

            # write the TOC entry
            tocentry[0] = (ext, n, i)
            fits['ALMTOC'].append(tocentry)

    logger.info('done with %d alms', len(alms))


def read_alms(filename, workdir='.', *, include=None, exclude=None):
    '''read a set of alms from a FITS file'''

    logger.info('reading alms from %s', filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of alms
    alms = {}

    # open the FITS file for reading
    with fitsio.FITS(path) as fits:

        # get the TOC from the FITS file
        fits_toc = fits['ALMTOC'].read()

        # read every entry in the TOC, add it to the list, then read the alms
        for entry in fits_toc:
            ext, n, i = entry[['EXT', 'NAME', 'BIN']]

            # skip if not selected
            if not toc_match((n, i), include=include, exclude=exclude):
                continue

            logger.info('reading %s alm for bin %s', n, i)

            # read the alm from the extension
            raw = fits[ext].read()
            alm = np.empty(len(raw), dtype=complex)
            alm.real = raw['real']
            alm.imag = raw['imag']
            del raw

            # read and attach metadata
            alm.dtype = np.dtype(alm.dtype, metadata=_read_metadata(fits[ext]))

            # store in set of alms
            alms[n, i] = alm

    logger.info('done with %d alms', len(alms))

    # return the dictionary of alms
    return alms


def write_cls(filename, cls, *, clobber=False, workdir='.', include=None, exclude=None):
    '''write a set of cls to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    '''

    logger.info('writing %d cls to %s', len(cls), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode='rw', clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode='rw', clobber=False) as fits:

        # write a new TOC extension if FITS doesn't already contain one
        if 'CLTOC' not in fits:
            fits.create_table_hdu(names=['EXT', 'NAME1', 'NAME2', 'BIN1', 'BIN2'],
                                  formats=['10A', '10A', '10A', 'I', 'I'],
                                  extname='CLTOC')

        # get a recarray to write TOC entries with
        tocentry = np.empty(1, dtype=fits['CLTOC'].get_rec_dtype()[0])

        # get the first free cl extension index
        cln = 0
        while f'CL{cln}' in fits:
            cln += 1

        # write every cl
        for (k1, k2, i1, i2), cl in cls.items():

            # skip if not selected
            if not toc_match((k1, k2, i1, i2), include=include, exclude=exclude):
                continue

            logger.info('writing %s x %s cl for bins %s, %s', k1, k2, i1, i2)

            # the cl extension name
            ext = f'CL{cln}'
            cln += 1

            # get the data into the binned format if not already
            if cl.dtype.names is None:
                dt = np.dtype([('L', float), ('CL', float),
                               ('LMIN', float), ('LMAX', float), ('W', float)],
                              metadata=dict(cl.dtype.metadata))
                cl_ = cl
                cl = np.empty(len(cl_), dt)
                cl['L'] = np.arange(len(cl_))
                cl['CL'] = cl_
                cl['LMIN'] = cl['L']
                cl['LMAX'] = cl['L'] + 1
                cl['W'] = 1

            # write the data column
            fits.write_table(cl, extname=ext)

            # write the metadata
            _write_metadata(fits[ext], cl.dtype.metadata)

            # write the TOC entry
            tocentry[0] = (ext, k1, k2, i1, i2)
            fits['CLTOC'].append(tocentry)

    logger.info('done with %d cls', len(cls))


def read_cls(filename, workdir='.', *, include=None, exclude=None):
    '''read a set of cls from a FITS file'''

    logger.info('reading cls from %s', filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of cls
    cls = {}

    # open the FITS file for reading
    with fitsio.FITS(path) as fits:

        # get the TOC from the FITS file
        fits_toc = fits['CLTOC'].read()

        # read every entry in the TOC, add it to the list, then read the cls
        for entry in fits_toc:
            ext, k1, k2, i1, i2 = entry[['EXT', 'NAME1', 'NAME2', 'BIN1', 'BIN2']]

            # skip if not selected
            if not toc_match((k1, k2, i1, i2), include=include, exclude=exclude):
                continue

            logger.info('reading %s x %s cl for bins %s, %s', k1, k2, i1, i2)

            # read the cl from the extension
            cl = fits[ext].read()

            # read and attach metadata
            cl.dtype = np.dtype(cl.dtype, metadata=_read_metadata(fits[ext]))

            # store in set of cls
            cls[k1, k2, i1, i2] = cl

    logger.info('done with %d cls', len(cls))

    # return the dictionary of cls
    return cls


def write_mms(filename, mms, *, clobber=False, workdir='.', include=None, exclude=None):
    '''write a set of mixing matrices to FITS file

    If the output file exists, the new mixing matrices will be appended, unless
    the ``clobber`` parameter is set to ``True``.

    '''

    logger.info('writing %d mm(s) to %s', len(mms), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode='rw', clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode='rw', clobber=False) as fits:

        # write a new TOC extension if FITS doesn't already contain one
        if 'MMTOC' not in fits:
            fits.create_table_hdu(names=['EXT', 'NAME', 'BIN1', 'BIN2'],
                                  formats=['10A', '10A', 'I', 'I'],
                                  extname='MMTOC')

        # get a recarray to write TOC entries with
        tocentry = np.empty(1, dtype=fits['MMTOC'].get_rec_dtype()[0])

        # get the first free mm extension index
        mmn = 0
        while f'MM{mmn}' in fits:
            mmn += 1

        # write every mixing matrix
        for (n, i1, i2), mm in mms.items():

            # skip if not selected
            if not toc_match((n, i1, i2), include=include, exclude=exclude):
                continue

            logger.info('writing mixing matrix %s for bins %s, %s', n, i1, i2)

            # the mm extension name
            ext = f'MM{mmn}'
            mmn += 1

            # write the mixing matrix as an image
            fits.write_image(mm, extname=ext)

            # write the WCS
            fits[ext].write_key('WCSAXES', 2)
            fits[ext].write_key('CNAME1', 'L_1')
            fits[ext].write_key('CNAME2', 'L_2')
            fits[ext].write_key('CTYPE1', ' ')
            fits[ext].write_key('CTYPE2', ' ')
            fits[ext].write_key('CUNIT1', ' ')
            fits[ext].write_key('CUNIT2', ' ')

            # write the metadata
            _write_metadata(fits[ext], mm.dtype.metadata)

            # write the TOC entry
            tocentry[0] = (ext, n, i1, i2)
            fits['MMTOC'].append(tocentry)

    logger.info('done with %d mm(s)', len(mms))


def read_mms(filename, workdir='.', *, include=None, exclude=None):
    '''read a set of mixing matrices from a FITS file'''

    logger.info('reading mixing matrices from %s', filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of mms
    mms = {}

    # open the FITS file for reading
    with fitsio.FITS(path) as fits:

        # get the TOC from the FITS file
        fits_toc = fits['MMTOC'].read()

        # read every entry in the TOC, add it to the list, then read the mms
        for entry in fits_toc:
            ext, n, i1, i2 = entry[['EXT', 'NAME', 'BIN1', 'BIN2']]

            # skip if not selected
            if not toc_match((n, i1, i2), include=include, exclude=exclude):
                continue

            logger.info('reading mixing matrix %s for bins %s, %s', n, i1, i2)

            # read the mixing matrix from the extension
            mm = fits[ext].read()

            # read and attach metadata
            mm.dtype = np.dtype(mm.dtype, metadata=_read_metadata(fits[ext]))

            # store in set of mms
            mms[n, i1, i2] = mm

    logger.info('done with %d mm(s)', len(mms))

    # return the dictionary of mms
    return mms


def write_cov(filename, cov, clobber=False, workdir='.', include=None, exclude=None):
    '''write a set of covariance matrices to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    '''

    logger.info('writing %d covariances to %s', len(cov), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode='rw', clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode='rw', clobber=False) as fits:

        # write a new TOC extension if FITS doesn't already contain one
        if 'COVTOC' not in fits:
            fits.create_table_hdu(
                names=['EXT', 'NAME1_1', 'NAME2_1', 'BIN1_1', 'BIN2_1',
                       'NAME1_2', 'NAME2_2', 'BIN1_2', 'BIN2_2'],
                formats=['10A', '10A', '10A', 'I', 'I', '10A', '10A', 'I', 'I'],
                extname='COVTOC')

        # get a recarray to write TOC entries with
        tocentry = np.empty(1, dtype=fits['COVTOC'].get_rec_dtype()[0])

        # get the first free cov extension index
        extn = 0
        while f'COV{extn}' in fits:
            extn += 1

        # write every covariance sub-matrix
        for (k1, k2), mat in cov.items():

            # skip if not selected
            if not toc_match((k1, k2), include=include, exclude=exclude):
                continue

            # the cl extension name
            ext = f'COV{extn}'
            extn += 1

            logger.info('writing %s x %s covariance matrix', k1, k2)

            # write the covariance matrix as an image
            fits.write_image(mat, extname=ext)

            # write the WCS
            fits[ext].write_key('WCSAXES', 2)
            fits[ext].write_key('CNAME1', 'L_1')
            fits[ext].write_key('CNAME2', 'L_2')
            fits[ext].write_key('CTYPE1', ' ')
            fits[ext].write_key('CTYPE2', ' ')
            fits[ext].write_key('CUNIT1', ' ')
            fits[ext].write_key('CUNIT2', ' ')

            # write the metadata
            _write_metadata(fits[ext], mat.dtype.metadata)

            # write the TOC entry
            tocentry[0] = (ext, *k1, *k2)
            fits['COVTOC'].append(tocentry)

    logger.info('done with %d covariance(s)', len(cov))


def read_cov(filename, workdir='.', *, include=None, exclude=None):
    '''read a set of covariances matrices from a FITS file'''

    logger.info('reading covariance matrices from %s', filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of covariances
    cov = {}

    # open the FITS file for reading
    with fitsio.FITS(path) as fits:

        # get the TOC from the FITS file
        fits_toc = fits['COVTOC'].read()

        # read every entry in the TOC, add it to the list, then read the data
        for entry in fits_toc:
            ext = entry['EXT']
            k1 = tuple(entry[['NAME1_1', 'NAME2_1', 'BIN1_1', 'BIN2_1']])
            k2 = tuple(entry[['NAME1_2', 'NAME2_2', 'BIN1_2', 'BIN2_2']])

            # skip if not selected
            if not toc_match((k1, k2), include=include, exclude=exclude):
                continue

            logger.info('reading %s x %s covariance matrix', k1, k2)

            # read the covariance matrix from the extension
            mat = fits[ext].read()

            # read and attach metadata
            mat.dtype = np.dtype(mat.dtype, metadata=_read_metadata(fits[ext]))

            # store in set
            cov[k1, k2] = mat

    logger.info('done with %d covariance(s)', len(cov))

    # return the toc dict of covariances
    return cov
