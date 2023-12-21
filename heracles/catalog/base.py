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
"""base definition for catalogue interface"""

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np


class CatalogPage:
    """One batch of rows from a catalogue.

    Internally holds all column data as a numpy array.

    """

    def _update(self):
        """Update internal data after dictionary changes."""
        # get and check size of rows
        size: int = -1
        for col, rows in self._data.items():
            if size == -1:
                size = len(rows)
            elif size != len(rows):
                msg = "inconsistent row length"
                raise ValueError(msg)
        self._size = size

    def __init__(self, data: Mapping) -> None:
        """Create a new catalogue page from given data."""
        self._data = {k: np.asanyarray(v) for k, v in data.items()}
        for v in self._data.values():
            v.flags.writeable = False
        self._update()

    def _getdata(self, key):
        """
        Return data for one single column.  Supports negating columns
        (``-COLUMN``), possibly more in the future.
        """
        if key[:1] == "-":
            return -self._data[key[1:]]
        return self._data[key]

    def __getitem__(self, col):
        """Return one or more columns without checking."""
        if isinstance(col, (list, tuple)):
            return tuple(self._getdata(c) for c in col)
        return self._getdata(col)

    def __len__(self):
        """Number of columns in the page."""
        return len(self._data)

    def __copy__(self):
        """Create a copy."""
        return self.copy()

    def __iter__(self):
        """Iterate over column names."""
        yield from self._data

    @property
    def names(self):
        """Column names in the page."""
        return list(self._data)

    @property
    def size(self):
        """Number of rows in the page."""
        return self._size

    @property
    def data(self):
        """Return an immutable view on the data of this page."""
        return MappingProxyType(self._data)

    def get(self, *col):
        """Return one or more columns with checking."""
        val = []
        for c in col:
            v = self._getdata(c)
            if np.any(np.isnan(v)):
                msg = f'invalid values in column "{c}"'
                raise ValueError(msg)
            val.append(v)
        if len(val) == 1:
            val = val[0]
        return val

    def copy(self) -> "CatalogPage":
        """Create new page instance with the same data."""
        return CatalogPage(self._data)

    def delete(self, where) -> None:
        """Delete the rows indicated by ``where``."""
        for col, rows in self._data.items():
            self._data[col] = np.delete(rows, where)
        self._update()


@runtime_checkable
class Catalog(Protocol):
    """protocol for catalogues"""

    def __getitem__(self, where):
        """create a view with the given selection"""
        ...

    @property
    def metadata(self):
        """return a dictionary of metadata for the catalogue"""
        ...

    @property
    def label(self):
        """return a human-friendly identifier for the source of data"""
        ...

    @property
    def base(self):
        """return the base catalogue of a view, or ``None`` if not a view"""
        ...

    @property
    def selection(self):
        """return the selection of a view, or ``None`` if not a view"""
        ...

    @property
    def names(self):
        """columns in the catalogue, or ``None`` if not known"""
        ...

    @property
    def size(self):
        """rows in the catalogue, or ``None`` if not known"""
        ...

    @property
    def visibility(self):
        """visibility map of the catalogue"""
        ...

    def where(self, selection, visibility=None):
        """create a view on this catalogue with the given selection"""
        ...

    @property
    def page_size(self):
        """page size for iteration"""
        ...

    def __iter__(self):
        """iterate over pages of rows in the catalogue"""
        ...

    def select(self, selection):
        """iterate over pages of rows with the given selection"""
        ...


class CatalogView:
    """a view of a catalogue with some selection applied"""

    def __init__(self, catalog, selection, visibility=None):
        """create a new view"""
        self._catalog = catalog
        self._selection = selection
        self._visibility = visibility

    def __repr__(self):
        """object representation of this view"""
        return f"{self._catalog!r}[{self._selection!r}]"

    def __str__(self):
        """string representation of this view"""
        return f"{self._catalog!s}[{self._selection!s}]"

    def __getitem__(self, where):
        """return a view with a subselection of this view"""
        return self.where(where)

    @property
    def base(self):
        """base catalogue of this view"""
        return self._catalog

    @property
    def metadata(self):
        """return metadata for this view"""
        return self._catalog.metadata

    @property
    def label(self):
        """human-friendly label of the catalogue (not settable in view)"""
        return self._catalog.label

    @property
    def selection(self):
        """selection of this view"""
        return self._selection

    @property
    def names(self):
        """column names of this view"""
        return self._catalog.names

    @property
    def size(self):
        """size of this view, might not take selection into account"""
        return self._catalog._size(self._selection)

    @property
    def visibility(self):
        """the visibility of this view"""
        if self._visibility is None:
            return self._catalog.visibility
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    def where(self, selection, visibility=None):
        """return a view with a subselection of this view"""
        if isinstance(selection, (tuple, list)):
            joined = (self._selection, *selection)
        else:
            joined = (self._selection, selection)
        if visibility is None:
            visibility = self._visibility
        return self._catalog.where(joined, visibility)

    @property
    def page_size(self):
        """page size for iterating this view"""
        return self._catalog.page_size

    def __iter__(self):
        """iterate the catalogue with the selection of this view"""
        yield from self._catalog.select(self._selection)

    def select(self, selection):
        """iterate over pages of rows with the given selection"""
        if isinstance(selection, (tuple, list)):
            joined = (self._selection, *selection)
        else:
            joined = (self._selection, selection)
        yield from self._catalog.select(joined)


class CatalogBase(metaclass=ABCMeta):
    """abstract base class for base catalogues (not views)"""

    default_page_size: int = 100_000
    """default value for page size"""

    def __init__(self):
        """Create a new catalogue instance."""

        self._page_size = self.default_page_size
        self._filters = []
        self._label = None
        self._visibility = None

    def __copy__(self):
        """return a shallow copy of the catalogue"""

        other = self.__class__.__new__(self.__class__)
        other._page_size = self._page_size
        other._filters = self._filters.copy()
        other._label = self._label
        other._visibility = self._visibility
        return other

    @abstractmethod
    def _names(self):
        """abstract method to return the columns in the catalogue"""
        ...

    @abstractmethod
    def _size(self, selection):
        """abstract method to return the size of the catalogue or selection"""
        ...

    @abstractmethod
    def _join(self, *where):
        """abstract method to join selections"""
        ...

    @abstractmethod
    def _pages(self, selection):
        """abstract method to iterate selected pages from the catalogue"""
        ...

    @property
    def filters(self):
        """filters to apply to this catalogue"""
        return self._filters

    @filters.setter
    def filters(self, filters):
        self._filters = filters

    def add_filter(self, filt):
        """add a filter to catalogue"""
        self.filters.append(filt)

    def __getitem__(self, where):
        """create a view on this catalogue with the given selection"""
        return self.where(where)

    @property
    def base(self):
        """returns ``None`` since this is not a view of another catalogue"""
        return

    @property
    def metadata(self):
        """return the metadata for this catalogue"""
        return MappingProxyType(
            {
                "catalog": self._label,
            },
        )

    @property
    def label(self):
        """optional human-friendly label for catalogue"""
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def selection(self):
        """returns ``None`` since this is not a view of another catalogue"""
        return

    @property
    def names(self):
        """columns in the catalogue, or ``None`` if not known"""
        return self._names()

    @property
    def size(self):
        """total rows in the catalogue, or ``None`` if not known"""
        return self._size(None)

    @property
    def visibility(self):
        """optional visibility map for catalogue"""
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    def where(self, selection, visibility=None):
        """create a view on this catalogue with the given selection"""
        if isinstance(selection, (tuple, list)):
            selection = self._join(*selection)
        return CatalogView(self, selection, visibility)

    @property
    def page_size(self):
        """number of rows per page (default: 100_000)"""
        return self._page_size

    @page_size.setter
    def page_size(self, value):
        self._page_size = value

    def __iter__(self):
        """iterate over pages of rows in the catalogue"""
        yield from self.select(None)

    def select(self, selection):
        """iterate over pages of rows with the given selection"""

        if isinstance(selection, (tuple, list)):
            selection = self._join(*selection)

        for page in self._pages(selection):
            # apply filters
            for filt in self._filters:
                filt(page)

            # yield the filtered page
            yield page
