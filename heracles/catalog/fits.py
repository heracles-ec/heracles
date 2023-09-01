# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023 Euclid Science Ground Segment
#
# This file is part of Heracles.
#
# Heracles is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Heracles is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Heracles. If not, see <https://www.gnu.org/licenses/>.
"""module for catalogue processing."""
import weakref

import fitsio

from .base import CatalogBase, CatalogPage


def _is_table_hdu(hdu):
    """Return true if HDU is a table with data."""
    return isinstance(hdu, fitsio.hdu.TableHDU) and hdu.has_data()


def rowfilter(array, expr):
    """Filter the rows of a structured array."""
    return eval(expr, None, {name: array[name] for name in array.dtype.names})


class FitsCatalog(CatalogBase):
    """flexible reader for catalogues from FITS files."""

    def __init__(self, filename, *, columns=None, ext=None) -> None:
        """Create a new FITS catalogue reader.

        Neither opens the FITS file nor reads the catalogue immediately.

        """
        super().__init__()
        self._filename = filename
        self._columns = columns
        self._ext = ext

    def __copy__(self):
        """Return a copy of this catalog."""
        other = super().__copy__()
        other._filename = self._filename
        other._columns = self._columns
        other._ext = self._ext
        return other

    def __repr__(self) -> str:
        """String representation of FitsCatalog."""
        s = self._filename
        if self._ext is not None:
            s = f"{s}[{self._ext!r}]"
        return s

    def hdu(self):
        """HDU for catalogue data."""
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
                        msg = "no table data in FITS"
                        raise TypeError(msg) from None
                else:
                    hdu = fits[self._ext]

            finally:
                # close fits if we didn't manage to get hdu
                if hdu is None:
                    fits.close()

            # make sure that internal _FITS is closed when hdu dies
            weakref.finalize(hdu, hdu._FITS.close)

            # cache hdu as a weak reference
            self._hdu = weakref.ref(hdu)

        return hdu

    def _names(self):
        """Column names in FITS catalogue."""
        # store column names on first access
        if self._columns is None:
            self._columns = self.hdu().get_colnames()
        return self._columns

    def _size(self, selection):
        """Size of FITS catalogue; selection is ignored."""
        return self.hdu().get_nrows()

    def _join(self, *where):
        """Join rowfilter expressions."""
        return (
            "(" + ") & (".join(map(str, filter(None, where))) + ")" if where else None
        )

    def _pages(self, selection):
        """Iterate pages of rows in FITS file, optionally using the query."""
        # keep an unchanging local copy of the page size
        page_size = self.page_size

        hdu = self.hdu()
        names = self._names()

        # use all rows or selection if one is given
        nrows = hdu.get_nrows()

        # information for caching
        hduid = id(hdu)

        # now iterate all rows in batches
        for start in range(0, nrows, page_size):
            stop = start + page_size

            # see if rows were cached
            try:
                rows = self._rows if self._rowinfo == (hduid, start, stop) else None
            except AttributeError:
                rows = None

            # retrieve rows if not cached
            if rows is None:
                rows = hdu[names][start:stop]

                # update row cache
                self._rowinfo = (hduid, start, stop)
                self._rows = rows

            # apply selection if given
            if selection is not None:
                rows = rows[rowfilter(rows, selection)]

            yield CatalogPage({name: rows[name] for name in names})
