'''module for catalogue processing'''

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from contextlib import contextmanager
from functools import lru_cache
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
        for v in self._data.values():
            v.flags.writeable = False
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


@runtime_checkable
class Catalog(Protocol):
    '''protocol for catalogues'''

    def __getitem__(self, where):
        '''create a view with the given selection'''
        ...

    @property
    def base(self):
        '''return the base catalogue of a view, or ``None`` if not a view'''
        ...

    @property
    def selection(self):
        '''return the selection of a view, or ``None`` if not a view'''
        ...

    @property
    def names(self):
        '''columns in the catalogue, or ``None`` if not known'''
        ...

    @property
    def size(self):
        '''rows in the catalogue, or ``None`` if not known'''
        ...

    @property
    def visibility(self):
        '''visibility map of the catalogue'''
        ...

    def where(self, selection, visibility=None):
        '''create a view on this catalogue with the given selection'''
        ...

    @property
    def page_size(self):
        '''page size for iteration'''
        ...

    def __iter__(self):
        '''iterate over pages of rows in the catalogue'''
        ...

    def select(self, selection):
        '''iterate over pages of rows with the given selection'''
        ...


class CatalogView:
    '''a view of a catalogue with some selection applied'''

    def __init__(self, catalog, selection, visibility=None):
        '''create a new view'''
        self._catalog = catalog
        self._selection = selection
        self._visibility = visibility

    def __repr__(self):
        '''object representation of this view'''
        return f'{self._catalog!r}[{self._selection!r}]'

    def __str__(self):
        '''string representation of this view'''
        return f'{self._catalog!s}[{self._selection!s}]'

    def __getitem__(self, where):
        '''return a view with a subselection of this view'''
        return self.where(where)

    @property
    def base(self):
        '''base catalogue of this view'''
        return self._catalog

    @property
    def selection(self):
        '''selection of this view'''
        return self._selection

    @property
    def names(self):
        '''column names of this view'''
        return self._catalog.names

    @property
    def size(self):
        '''size of this view, might not take selection into account'''
        return self._catalog._size(self._selection)

    @property
    def visibility(self):
        '''the visibility of this view'''
        if self._visibility is None:
            return self._catalog.visibility
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    def where(self, selection, visibility=None):
        '''return a view with a subselection of this view'''
        if isinstance(selection, (tuple, list)):
            joined = (self._selection, *selection)
        else:
            joined = (self._selection, selection)
        if visibility is None:
            visibility = self._visibility
        return self._catalog.where(joined, visibility)

    @property
    def page_size(self):
        '''page size for iterating this view'''
        return self._catalog.page_size

    def __iter__(self):
        '''iterate the catalogue with the selection of this view'''
        yield from self._catalog.select(self._selection)

    def select(self, selection):
        '''iterate over pages of rows with the given selection'''
        if isinstance(selection, (tuple, list)):
            joined = (self._selection, *selection)
        else:
            joined = (self._selection, selection)
        yield from self._catalog.select(joined)


class CatalogBase(metaclass=ABCMeta):
    '''abstract base class for base catalogues (not views)'''

    default_page_size: int = 100_000
    '''default value for page size'''

    def __init__(self):
        '''Create a new catalogue instance.'''

        self._page_size = self.default_page_size
        self._filters = []
        self._visibility = None

    def __copy__(self):
        '''return a shallow copy of the catalogue'''

        other = self.__class__.__new__(self.__class__)
        other._page_size = self._page_size
        other._filters = self._filters.copy()
        other._visibility = self._visibility
        return other

    @abstractmethod
    def _names(self):
        '''abstract method to return the columns in the catalogue'''
        ...

    @abstractmethod
    def _size(self, selection):
        '''abstract method to return the size of the catalogue or selection'''
        ...

    @abstractmethod
    def _join(self, *where):
        '''abstract method to join selections'''
        ...

    @abstractmethod
    def _pages(self, selection):
        '''abstract method to iterate selected pages from the catalogue'''
        ...

    @property
    def filters(self):
        '''filters to apply to this catalogue'''
        return self._filters

    @filters.setter
    def filters(self, filters):
        self._filters = filters

    def add_filter(self, filt):
        '''add a filter to catalogue'''
        self.filters.append(filt)

    def __getitem__(self, where):
        '''create a view on this catalogue with the given selection'''
        return self.where(where)

    @property
    def base(self):
        '''returns ``None`` since this is not a view of another catalogue'''
        return None

    @property
    def selection(self):
        '''returns ``None`` since this is not a view of another catalogue'''
        return None

    @property
    def names(self):
        '''columns in the catalogue, or ``None`` if not known'''
        return self._names()

    @property
    def size(self):
        '''total rows in the catalogue, or ``None`` if not known'''
        return self._size(None)

    @property
    def visibility(self):
        '''optional visibility map for catalogue'''
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    def where(self, selection, visibility=None):
        '''create a view on this catalogue with the given selection'''
        if isinstance(selection, (tuple, list)):
            selection = self._join(*selection)
        return CatalogView(self, selection, visibility)

    @property
    def page_size(self):
        '''number of rows per page (default: 100_000)'''
        return self._page_size

    @page_size.setter
    def page_size(self, value):
        self._page_size = value

    def __iter__(self):
        '''iterate over pages of rows in the catalogue'''
        yield from self.select(None)

    def select(self, selection):
        '''iterate over pages of rows with the given selection'''

        if isinstance(selection, (tuple, list)):
            selection = self._join(*selection)

        for page in self._pages(selection):

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

    def __repr__(self):
        name = self.__class__.__name__
        args = list(map(repr, self.columns))
        args += [f'weight={self.weight!r}', f'warn={self.warn!r}']
        args = ', '.join(args)
        return f'{name}({args})'

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

    def __repr__(self):
        name = self.__class__.__name__
        lon, lat = self.lonlat
        return f'{name}(..., {lon!r}, {lat!r})'

    def __call__(self, page):
        '''filter catalog page'''

        lon, lat = self._lonlat
        ipix = hp.ang2pix(self._nside, page[lon], page[lat], lonlat=True)
        exclude = np.where(self._footprint[ipix] == 0)[0]
        page.delete(exclude)


class ArrayCatalog(CatalogBase):
    '''catalogue reader for arrays'''

    def __init__(self, arr):
        '''create a new array catalogue reader'''
        super().__init__()
        self._arr = arr

    def __copy__(self):
        '''return a copy of this catalogue'''
        other = super().__copy__()
        other._arr = self._arr
        return other

    def _names(self):
        return self._arr.dtype.names

    def _size(self, selection):
        if selection is None:
            return len(self._arr)
        else:
            return len(self._arr[selection])

    def _join(self, first, *other):
        '''join boolean masks'''
        mask = first
        for a in other:
            mask = mask & a
        return mask

    def _pages(self, selection):
        '''iterate the rows of the array in pages'''
        if selection is None:
            arr = self._arr
        else:
            arr = self._arr[selection]
        nrows = len(arr)
        page_size = self.page_size
        names = arr.dtype.names
        for i in range(0, nrows, page_size):
            page = {name: arr[name][i:i+page_size] for name in names}
            yield CatalogPage(page)


def _is_table_hdu(hdu):
    '''return true if HDU is a table with data'''
    return isinstance(hdu, fitsio.hdu.TableHDU) and hdu.has_data()


class FitsCatalog(CatalogBase):
    '''flexible reader for catalogues from FITS files'''

    def __init__(self, filename, *, columns=None, ext=None):
        '''create a new FITS catalogue reader

        Neither opens the FITS file nor reads the catalogue immediately.

        '''
        super().__init__()
        self._filename = filename
        self._columns = columns
        self._ext = ext

    def __copy__(self):
        '''return a copy of this catalog'''
        other = super().__copy__()
        other._filename = self._filename
        other._columns = self._columns
        other._ext = self._ext
        return other

    def __repr__(self):
        '''string representation of FitsCatalog'''
        s = self._filename
        if self._ext is not None:
            s = s + f'[{self._ext!r}]'
        return s

    @contextmanager
    def hdu(self):
        '''context manager to get the HDU for catalogue data'''

        # use FITS as inner context manager to ensure file is closed
        with fitsio.FITS(self._filename) as fits:
            if self._ext is None:
                try:
                    # find table data extension
                    hdu = next(filter(_is_table_hdu, fits))
                except StopIteration:
                    raise TypeError('no extension with table data') from None
            else:
                hdu = fits[self._ext]
            try:
                yield hdu
            finally:
                pass

    def _names(self):
        '''column names in FITS catalogue'''
        # store column names on first access
        if self._columns is None:
            with self.hdu() as hdu:
                self._columns = hdu.get_colnames()
        return self._columns

    @lru_cache
    def _size(self, selection):
        '''size of FITS catalogue or selection'''
        with self.hdu() as hdu:
            if selection is None:
                return hdu.get_nrows()
            else:
                return len(hdu.where(selection))

    def _join(self, *where):
        '''join rowfilter expressions'''
        if not where:
            return 'true'
        return '(' + ') && ('.join(map(str, where)) + ')'

    def _pages(self, selection):
        '''iterate pages of rows in FITS file, optionally using the query'''

        # keep an unchanging local copy of the page size
        page_size = self.page_size

        with self.hdu() as hdu:

            names = self._names()

            # use all rows or selection if one is given
            if selection is None:
                selected = None
                size = hdu.get_nrows()
            else:
                selected = hdu.where(selection)
                size = len(selected)

            # now iterate the (selected) rows in batches
            for i in range(0, size, page_size):
                if selected is None:
                    rows = hdu[names][i:i+page_size]
                else:
                    rows = hdu[names][selected[i:i+page_size]]
                yield CatalogPage({name: rows[name] for name in names})
