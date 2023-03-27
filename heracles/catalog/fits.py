'''module for catalogue processing'''

from functools import lru_cache
from weakref import ref, finalize
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

    def hdu(self):
        '''HDU for catalogue data'''

        # see if there's a reference to hdu still around
        try:
            hdu = self._hdu()
        except AttributeError:
            hdu = None

        # if there is no cached HDU, open it
        if hdu is None:
            # need to open the fits file explicitly, not via context manager
            # we will not close it, to keep the HDU alive
            fits = fitsio.FITS(self._filename)

            # but ensure fits gets closed in case of error
            try:

                # get HDU from the file
                if self._ext is None:
                    try:
                        # find table data extension
                        hdu = next(filter(_is_table_hdu, fits))
                    except StopIteration:
                        raise TypeError('no table data in FITS') from None
                else:
                    hdu = fits[self._ext]

            finally:

                # close fits if we didn't manage to get hdu
                if hdu is None:
                    fits.close()

            # make sure that internal _FITS is closed when hdu dies
            finalize(hdu, hdu._FITS.close)

            # cache hdu as a weak reference
            self._hdu = ref(hdu)

        return hdu

    def _names(self):
        '''column names in FITS catalogue'''
        # store column names on first access
        if self._columns is None:
            self._columns = self.hdu().get_colnames()
        return self._columns

    @lru_cache
    def _size(self, selection):
        '''size of FITS catalogue or selection'''
        if selection is None:
            return self.hdu().get_nrows()
        else:
            return len(self.hdu().where(selection))

    def _join(self, *where):
        '''join rowfilter expressions'''
        if not where:
            return 'true'
        return '(' + ') && ('.join(map(str, where)) + ')'

    def _pages(self, selection):
        '''iterate pages of rows in FITS file, optionally using the query'''

        # keep an unchanging local copy of the page size
        page_size = self.page_size

        hdu = self.hdu()
        names = self._names()

        # use all rows or selection if one is given
        if selection is None:
            selected = None
            size = hdu.get_nrows()
        else:
            selected = hdu.where(selection)
            size = len(selected)

        # information for caching
        hduid = id(hdu)

        # now iterate the (selected) rows in batches
        for start in range(0, size, page_size):
            stop = start + page_size

            # see if page was cached
            try:
                if self._pageinfo == (hduid, start, stop):
                    page = self._page()
                else:
                    page = None
            except AttributeError:
                page = None

            # retrieve page if not cached
            if page is None:
                if selected is None:
                    rows = hdu[names][start:stop]
                else:
                    rows = hdu[names][selected[start:stop]]

                page = CatalogPage({name: rows[name] for name in names})

                self._pageinfo = (hduid, start, stop)
                self._page = ref(page)

            yield page
