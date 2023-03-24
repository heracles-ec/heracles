'''module for catalogue processing'''

from functools import lru_cache
from contextlib import contextmanager
import fitsio

from .base import CatalogBase, CatalogPage


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
