'''module for catalogue processing'''

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import partial
from copy import copy
import logging

import numpy as np
import healpy as hp
import fitsio

logger = logging.getLogger(__name__)


CatalogRows = namedtuple('CatalogRows', ['size', 'ra', 'dec', 'g1', 'g2', 'w'])
CatalogRows.__doc__ += ': selected columns from rows of catalogue data'


CatalogColumns = namedtuple('CatalogColumns', ['ra', 'dec', 'g1', 'g2', 'w'])
CatalogColumns.__doc__ += ': catalogue column definition'


def delete_catalog_rows(catalog_rows, where):
    '''delete selected rows from a CatalogRows tuple'''
    cols = catalog_rows._fields[1:]
    repl = {}
    for col in cols:
        val = getattr(catalog_rows, col)
        if val is not None:
            val = np.delete(val, where)
        repl[col] = val
    size = catalog_rows.size - np.size(where)
    return CatalogRows(size=size, **repl)


class Catalog(metaclass=ABCMeta):
    '''abstract base class for catalogues'''

    default_batch_size = 100_000
    '''default value for batch size'''

    def __init__(self, *, footprint=None):
        '''create a new catalogue instance'''

        self._footprint = footprint
        self._batch_size = self.default_batch_size
        self._conjugate_shear = False
        self._allow_invalid_positions = False
        self._allow_invalid_shears = False

    @property
    def footprint(self):
        '''optional footprint to apply at runtime'''
        return self._footprint

    @footprint.setter
    def footprint(self, footprint):
        self._footprint = footprint

    @property
    def batch_size(self):
        '''number of rows per batch (default: 100_000)'''
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def conjugate_shear(self):
        '''conjugate shear by flipping the sign of g2 (default: False)'''
        return self._conjugate_shear

    @conjugate_shear.setter
    def conjugate_shear(self, value):
        self._conjugate_shear = value

    @property
    def allow_invalid_positions(self):
        '''allow invalid values in the position columns (default: False)'''
        return self._allow_invalid_positions

    @allow_invalid_positions.setter
    def allow_invalid_positions(self, value):
        self._allow_invalid_positions = value

    @property
    def allow_invalid_shears(self):
        '''allow invalid values in the shears columns (default: False)'''
        return self._allow_invalid_shears

    @allow_invalid_shears.setter
    def allow_invalid_shears(self, value):
        self._allow_invalid_shears = value

    @abstractmethod
    def _rows(self):
        '''abstract method to retrieve batches of rows in the catalogue'''
        while False:
            yield None

    def __iter__(self):
        '''iterate over batches of rows in the catalogue'''

        # keep an unchanging local copy of the footprint while iterating
        if self._footprint is not None:
            footprint = self._footprint
            nside_fp = hp.get_nside(footprint)
        else:
            footprint = None

        # keep track of warnings having been emitted
        warned_pos = False
        warned_she = False

        # iterate the rows as returned by the concrete implementation
        # perform common post-processing:
        # - check positions are provided
        # - check invalid values
        # - conjugate shears
        # - apply footprint if given
        for rows in self._rows():

            # skip empty results
            if rows.size == 0:
                continue

            # check positions are given
            if rows.ra is None or rows.dec is None:
                raise TypeError('catalog did not yield position columns (ra, dec)')

            # check either both or none of g1, g2 are given
            if (rows.g1 is None) != (rows.g2 is None):
                raise TypeError('catalog yielded half a shear (only g1 or g2)')

            # check for invalid positions
            invalid = np.where(np.isnan(rows.ra) | np.isnan(rows.dec))[0]
            if len(invalid) > 0:
                if self._allow_invalid_positions:
                    if not warned_pos:
                        logger.warning('WARNING: catalog contains invalid positions')
                        warned_pos = True
                else:
                    raise ValueError('catalog contains invalid positions -- '
                                     'set allow_invalid_positions = True if you want '
                                     'to ignore the invalid values, but note that this '
                                     'changes your source distribution')
                rows = delete_catalog_rows(rows, invalid)
            del invalid

            # perform actions on shears if present
            if rows.g1 is not None:

                # check for invalid shears
                # invalid shears with weight 0 are silently deleted
                invalid = np.where(np.isnan(rows.g1) | np.isnan(rows.g2))[0]
                if len(invalid) > 0:
                    if rows.w is None or np.any(rows.w[invalid] != 0):
                        if self._allow_invalid_shears:
                            if not warned_she:
                                logger.warning('WARNING: catalog contains invalid shears')
                                warned_she = True
                        else:
                            raise ValueError('catalog contains invalid shears -- '
                                             'set allow_invalid_shears = True if you want '
                                             'to ignore the invalid values, but note that this '
                                             'changes your source distribution')
                    rows = delete_catalog_rows(rows, invalid)
                del invalid

                # conjugate shears if asked to
                # don't modify in place here
                if self._conjugate_shear:
                    rows = rows._replace(g2=-rows.g2)

            # exclude positions by footprint if given
            if footprint is not None:
                ipix = hp.ang2pix(nside_fp, rows.ra, rows.dec, lonlat=True)
                exclude = np.where(footprint[ipix] == 0)[0]
                rows = delete_catalog_rows(rows, exclude)
                del ipix, exclude

            # yield the post-processed rows
            yield rows


def column_reader(rows, *, ra=None, dec=None, g1=None, g2=None, w=None):
    '''select given columns from rows of catalogue data'''
    ra_ = rows[ra] if ra is not None else None
    dec_ = rows[dec] if dec is not None else None
    g1_ = rows[g1] if g1 is not None else None
    g2_ = rows[g2] if g2 is not None else None
    w_ = rows[w] if w is not None else None
    return CatalogRows(size=len(rows), ra=ra_, dec=dec_, g1=g1_, g2=g2_, w=w_)


def _make_columns(columns):
    '''create a column reader from various definitions'''

    if isinstance(columns, tuple):
        ra, dec, g1, g2, w = columns
        reader = partial(column_reader, ra=ra, dec=dec, g1=g1, g2=g2, w=w)
    elif callable(columns):
        reader = columns
    else:
        raise TypeError('columns must be a CatalogColumns instance, '
                        'a tuple (ra, dec, g1, g2, w) of column names, '
                        'or a callable')

    return reader


def _array_columns(rows):
    '''create columns from an unstructured array'''
    nrows, ncols, *_ = np.shape(rows)
    ra = rows[:, 0] if ncols > 0 else None
    dec = rows[:, 1] if ncols > 1 else None
    g1 = rows[:, 2] if ncols > 2 else None
    g2 = rows[:, 3] if ncols > 3 else None
    w = rows[:, 4] if ncols > 4 else None
    return CatalogRows(nrows, ra=ra, dec=dec, g1=g1, g2=g2, w=w)


class ArrayCatalog(Catalog):
    '''catalogue reader for arrays'''

    def __init__(self, arr, columns=None, *, footprint=None):
        '''create a new array catalogue reader'''

        super().__init__(footprint=footprint)

        self._arr = arr
        if columns is None:
            self._columns = _array_columns
        else:
            self._columns = _make_columns(columns)

    def _rows(self):
        '''iterate the rows of the array in batches'''

        arr = self._arr
        nrows = len(arr)
        batch_size = self.batch_size
        columns = self._columns
        for i in range(0, nrows, batch_size):
            rows = arr[i:i+batch_size]
            yield columns(rows)


def _fits_table_hdu(fits):
    '''get the first FITS extension with table data'''
    try:
        hdu = next(hdu for hdu in fits if hdu.has_data())
    except StopIteration:
        raise TypeError('cannot find FITS extension with table data') from None
    return hdu


class FitsCatalog(Catalog):
    '''flexible reader for catalogues from FITS files'''

    def __init__(self, filename, columns, *, ext=None, query=None, footprint=None):
        '''create a new FITS catalogue reader

        Neither opens the FITS file nor reads the catalogue immediately.

        '''

        super().__init__(footprint=footprint)

        self._filename = filename
        self._ext = ext
        self._query = query
        self._columns = _make_columns(columns)

    def __repr__(self):
        '''string representation of FitsCatalog'''
        s = self._filename
        if self._ext is not None:
            s = s + f'[{self._ext!r}]'
        if self._query is not None:
            s = s + f'[{self._query!r}]'
        return s

    def query(self, query):
        '''return a new FitsCatalog instance with an additional query'''

        copied = copy(self)

        if self._query is None:
            copied._query = query
        else:
            copied._query = f'({self._query}) && ({query})'

        return copied

    def _rows(self):
        '''iterate rows in FITS file, optionally using the query'''

        with fitsio.FITS(self._filename) as fits:

            # find or get the extension
            if self._ext is None:
                hdu = _fits_table_hdu(fits)
            else:
                hdu = fits[self._ext]

            # use all rows or select from query if one is given
            if self._query is None:
                selected = None
                nrows = hdu.get_nrows()
            else:
                selected = hdu.where(self._query)
                nrows = len(selected)

            # keep an unchanging local copy of the batch size and columns
            batch_size = self.batch_size
            columns = self._columns

            # now iterate the (selected) rows in batches
            for i in range(0, nrows, batch_size):
                if selected is None:
                    rows = hdu[i:i+batch_size]
                else:
                    rows = hdu[selected[i:i+batch_size]]
                yield columns(rows)
