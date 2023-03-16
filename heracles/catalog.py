'''module for catalogue processing'''

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from copy import copy
import warnings

import numpy as np
import healpy as hp
import fitsio


class CatalogPage:
    '''One batch of rows from a catalogue.

    Internally holds all column data as a numpy array.

    '''

    def _update(self):
        '''Update internal data after dictionary changes.'''
        # get and check size of rows
        size: int = -1
        for col, rows in self._data.items():
            if size == -1:
                size = len(rows)
            elif size != len(rows):
                raise ValueError('inconsistent row length')
        self._size = size

    def __init__(self, data: Mapping) -> None:
        '''Create a new catalogue page from given data.'''
        self._data = {k: np.asanyarray(v) for k, v in data.items()}
        self._update()

    def __getitem__(self, col):
        '''Return one or more columns without checking.'''
        if isinstance(col, (list, tuple)):
            return tuple(self._data[c] for c in col)
        else:
            return self._data[col]

    def __len__(self):
        '''Number of columns in the page.'''
        return len(self._data)

    def __copy__(self):
        '''Create a copy.'''
        return self.copy()

    def __iter__(self):
        '''Iterate over column names.'''
        yield from self._data

    @property
    def names(self):
        '''Column names in the page.'''
        return list(self._data)

    @property
    def size(self):
        '''Number of rows in the page.'''
        return self._size

    @property
    def data(self):
        '''Return an immutable view on the data of this page.'''
        return MappingProxyType(self._data)

    def get(self, *col):
        '''Return one or more columns with checking.'''
        val = []
        for c in col:
            v = self._data[c]
            if np.any(np.isnan(v)):
                raise ValueError(f'invalid values in column "{c}"')
            val.append(v)
        if len(val) == 1:
            val = val[0]
        return val

    def copy(self) -> 'CatalogPage':
        '''Create new page instance with the same data.'''
        return CatalogPage(self._data)

    def delete(self, where) -> None:
        '''Delete the rows indicated by ``where``.'''
        for col, rows in self._data.items():
            self._data[col] = np.delete(rows, where)
        self._update()


class Catalog(metaclass=ABCMeta):
    '''abstract base class for catalogues'''

    default_page_size: int = 100_000
    '''default value for page size'''

    def __init__(self):
        '''Create a new catalogue instance.'''

        self._page_size = self.default_page_size
        self._filters = []
        self._visibility = None
        self._names = None
        self._size = None

    def __copy__(self):
        '''return a shallow copy of the catalogue'''

        other = self.__class__.__new__(self.__class__)

        other._page_size = self._page_size
        other._filters = self._filters.copy()
        other._visibility = self._visibility
        other._names = self._names
        other._size = self._size

        return other

    @property
    def page_size(self):
        '''number of rows per page (default: 100_000)'''
        return self._page_size

    @page_size.setter
    def page_size(self, value):
        self._page_size = value

    @property
    def filters(self):
        '''filters to apply to this catalogue'''
        return self._filters

    @filters.setter
    def filters(self, filters):
        self._filters = filters

    @property
    def visibility(self):
        '''optional visibility map for catalogue'''
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    @property
    def names(self):
        '''columns in the catalogue, if known, or None'''
        return self._names

    @property
    def size(self):
        '''total rows in the catalogue, if known, or None'''
        return self._size

    def add_filter(self, filt):
        '''add a filter to catalogue'''
        self.filters.append(filt)

    @abstractmethod
    def _pages(self):
        '''abstract method to retrieve pages of rows from the catalogue'''
        ...

    def __iter__(self):
        '''iterate over pages of rows in the catalogue'''

        for page in self._pages():

            # skip empty pages
            if page.size == 0:
                continue

            # apply filters
            for filt in self._filters:
                filt(page)

            # yield the filtered page
            yield page


class InvalidValueFilter:
    '''Filter invalid values from a catalogue.'''

    def __init__(self, *columns, weight=None, warn=True):
        '''Filter invalid values in the given columns.

        If ``warn`` is true, invalid values will emit a warning.

        '''

        self.columns = columns
        self.weight = weight
        self.warn = warn

    def __call__(self, page):
        '''Filter a catalog page.'''

        invalid_mask = np.zeros(page.size, dtype=bool)
        for col in self.columns:
            invalid_mask |= np.isnan(page[col])
        if self.weight is not None:
            invalid_mask &= (page[self.weight] != 0)
        invalid = np.where(invalid_mask)[0]
        if len(invalid) > 0:
            if self.warn:
                warnings.warn('WARNING: catalog contains invalid values')
            page.delete(invalid)


class FootprintFilter:
    '''Filter a catalogue using a footprint map.'''

    def __init__(self, footprint, lon, lat):
        '''Filter using the given footprint map and position columns.'''
        self._footprint = footprint
        self._nside = hp.get_nside(footprint)
        self._lonlat = (lon, lat)

    @property
    def footprint(self):
        '''footprint for filter'''
        return self._footprint

    @property
    def lonlat(self):
        '''longitude and latitude columns'''
        return self._lonlat

    def __call__(self, page):
        '''filter catalog page'''

        lon, lat = self._lonlat
        ipix = hp.ang2pix(self._nside, page[lon], page[lat], lonlat=True)
        exclude = np.where(self._footprint[ipix] == 0)[0]
        page.delete(exclude)


class ArrayCatalog(Catalog):
    '''catalogue reader for arrays'''

    def __init__(self, arr):
        '''create a new array catalogue reader'''
        super().__init__()
        self._arr = arr
        self._size = len(arr)
        self._names = arr.dtype.names

    def __copy__(self):
        '''return a copy of this catalogue'''
        other = super().__copy__()
        other._arr = self._arr
        return other

    def _pages(self):
        '''iterate the rows of the array in pages'''
        arr = self._arr
        nrows = len(arr)
        page_size = self.page_size
        names = arr.dtype.names
        for i in range(0, nrows, page_size):
            page = {name: arr[name][i:i+page_size] for name in names}
            yield CatalogPage(page)


def _fits_table_hdu(fits):
    '''get the first FITS extension with table data'''
    try:
        hdu = next(hdu for hdu in fits if hdu.has_data())
    except StopIteration:
        raise TypeError('cannot find FITS extension with table data') from None
    return hdu


class FitsCatalog(Catalog):
    '''flexible reader for catalogues from FITS files'''

    def __init__(self, filename, *, columns=None, ext=None, query=None):
        '''create a new FITS catalogue reader

        Neither opens the FITS file nor reads the catalogue immediately.

        '''

        super().__init__()

        self._filename = filename
        self._columns = columns
        self._ext = ext
        self._query = query

    def __copy__(self):
        '''return a copy of this catalog'''
        other = super().__copy__()
        other._filename = self._filename
        other._columns = self._columns
        other._ext = self._ext
        other._query = self._query
        return other

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

        copied._size = None

        return copied

    def _open(self, fits):
        '''open FITS for reading'''

        # find or get the extension
        if self._ext is None:
            hdu = _fits_table_hdu(fits)
        else:
            hdu = fits[self._ext]

        # use all columns or the selected ones
        if self._columns is None:
            names = hdu.get_colnames()
        else:
            names = self._columns

        # use all rows or select from query if one is given
        if self._query is None:
            selected = None
            size = hdu.get_nrows()
        else:
            selected = hdu.where(self._query)
            size = len(selected)

        return hdu, names, selected, size

    def peek(self):
        '''read the FITS file information'''
        with fitsio.FITS(self._filename) as fits:
            _, names, _, size = self._open(fits)
            self._size = size
            self._names = names

    def _pages(self):
        '''iterate pages of rows in FITS file, optionally using the query'''

        # keep an unchanging local copy of the page size
        page_size = self.page_size

        with fitsio.FITS(self._filename) as fits:

            hdu, names, selected, nrows = self._open(fits)

            # set catalogue info
            self._size = nrows
            self._names = names

            # now iterate the (selected) rows in batches
            for i in range(0, nrows, page_size):
                if selected is None:
                    rows = hdu[names][i:i+page_size]
                else:
                    rows = hdu[names][selected[i:i+page_size]]
                yield CatalogPage({name: rows[name] for name in names})
