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
"""module for field definitions"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import combinations_with_replacement, product
from typing import TYPE_CHECKING

import coroutines
import numpy as np

from .core import toc_match, update_metadata

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Mapping, Sequence
    from typing import TypeGuard

    from numpy.typing import ArrayLike

    from .catalog import Catalog, CatalogPage
    from .maps import Mapper
    from .progress import ProgressTask


# type alias for column specification
Columns = tuple["str | None", ...]


class Field(metaclass=ABCMeta):
    """Abstract base class for field definitions.

    Concrete classes must implement the `__call__()` method which takes
    a catalogue instance and returns a coroutine for mapping.

    """

    # column names: "col1", "col2", "[optional]"
    uses: Sequence[str] | str | None = None

    # every field subclass has a static spin weight attribute, which can be
    # overwritten by the class (or even an individual instance)
    __spin: int | None = None

    # definition of required and optional columns
    __ncol: tuple[int, int]

    def __init_subclass__(cls, *, spin: int | None = None) -> None:
        """Initialise spin weight of field subclasses."""
        super().__init_subclass__()
        if spin is not None:
            cls.__spin = spin
        uses = cls.uses
        if uses is None:
            uses = ()
        elif isinstance(uses, str):
            uses = (uses,)
        ncol = len(uses)
        nopt = 0
        for u in uses[::-1]:
            if u.startswith("[") and u.endswith("]"):
                nopt += 1
            else:
                break
        cls.__ncol = (ncol - nopt, ncol)

    def __init__(
        self,
        mapper: Mapper | None,
        *columns: str,
        mask: str | None = None,
    ) -> None:
        """Initialise the field."""
        super().__init__()
        self.__mapper = mapper
        self.__columns = self._init_columns(*columns) if columns else None
        self.__mask = mask

    @classmethod
    def _init_columns(cls, *columns: str) -> Columns:
        """Initialise the given set of columns for a specific field
        subclass."""
        nmin, nmax = cls.__ncol
        if not nmin <= len(columns) <= nmax:
            uses = cls.uses
            if uses is None:
                uses = ()
            if isinstance(uses, str):
                uses = (uses,)
            count = f"{nmin}"
            if nmax != nmin:
                count += f" to {nmax}"
            msg = f"field of type '{cls.__name__}' accepts {count} columns"
            if uses:
                msg += " (" + ", ".join(uses) + ")"
            msg += f", received {len(columns)}"
            raise ValueError(msg)
        return columns + (None,) * (nmax - len(columns))

    @property
    def mapper(self) -> Mapper | None:
        """Return the mapper used by this field."""
        return self.__mapper

    @property
    def mapper_or_error(self) -> Mapper:
        """Return the mapper used by this field, or raise a :class:`ValueError`
        if not set."""
        if self.__mapper is None:
            msg = "no mapper for field"
            raise ValueError(msg)
        return self.__mapper

    @property
    def columns(self) -> Columns | None:
        """Return the catalogue columns used by this field."""
        return self.__columns

    @property
    def columns_or_error(self) -> Columns:
        """Return the catalogue columns used by this field, or raise a
        :class:`ValueError` if not set."""
        if self.__columns is None:
            msg = "no columns for field"
            raise ValueError(msg)
        return self.__columns

    @property
    def spin(self) -> int:
        """Spin weight of field."""
        spin = self.__spin
        if spin is None:
            clsname = self.__class__.__name__
            msg = f"field of type '{clsname}' has undefined spin weight"
            raise ValueError(msg)
        return spin

    @property
    def mask(self) -> str | None:
        """Name of the mask for this field."""
        return self.__mask

    @abstractmethod
    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Implementation for mapping a catalogue."""
        ...


async def _pages(
    catalog: Catalog,
    progress: ProgressTask | None,
) -> AsyncIterable[CatalogPage]:
    """
    Asynchronous generator for the pages of a catalogue.  Also manages
    progress updates.
    """
    page_size = catalog.page_size
    if progress:
        progress.update(completed=0, total=catalog.size)
    for page in catalog:
        await coroutines.sleep()
        yield page
        if progress:
            progress.update(advance=page_size)
    # suspend again to give all concurrent loops a chance to finish
    await coroutines.sleep()


class Positions(Field, spin=0):
    """Field of positions in a catalogue.

    Can produce both overdensity maps and number count maps, depending
    on the ``overdensity`` property.

    """

    uses = "longitude", "latitude"

    def __init__(
        self,
        *columns: str,
        overdensity: bool = True,
        nbar: float | None = None,
        mask: str | None = None,
    ) -> None:
        """Create a position field."""
        super().__init__(*columns, mask=mask)
        self.__overdensity = overdensity
        self.__nbar = nbar

    @property
    def overdensity(self) -> bool:
        """Flag to create overdensity maps."""
        return self.__overdensity

    @property
    def nbar(self) -> float | None:
        """Mean number count."""
        return self.__nbar

    @nbar.setter
    def nbar(self, nbar: float | None) -> None:
        """Set the mean number count."""
        self.__nbar = nbar

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map the given catalogue."""

        # get mapper
        mapper = self.mapper_or_error

        # get catalogue column definition
        col = self.columns_or_error

        # position map
        pos = mapper.create(spin=self.spin)

        # keep track of the total number of galaxies
        ngal = 0

        # map catalogue data asynchronously
        async for page in _pages(catalog, progress):
            lon, lat = page.get(*col)
            mapper.map_values(lon, lat, [pos])

            ngal += page.size

            # clean up to free unneeded memory
            del page, lon, lat

        # get visibility map if present in catalogue
        vmap = catalog.visibility

        # match resolution of visibility map if present
        # FIXME generic mapper support
        if vmap is not None and vmap.size != pos.size:
            import healpy as hp

            warnings.warn("position and visibility maps have different NSIDE")
            vmap = hp.ud_grade(vmap, mapper.nside)

        # mean visibility (i.e. f_sky)
        if vmap is None:
            vbar = 1
        else:
            vbar = np.mean(vmap)

        # effective number of pixels
        npix = 4 * np.pi / mapper.area

        # compute average number count from map
        nbar = ngal / vbar / npix
        # override with provided value, but check that it makes sense
        if (nbar_ := self.nbar) is not None:
            # Poisson std dev from expected ngal assuming nbar_ is truth
            sigma_nbar = (nbar_ / vbar / npix) ** 0.5
            if abs(nbar - nbar_) > 3 * sigma_nbar:
                warnings.warn(
                    f"The provided mean density ({nbar_:g}) differs from the "
                    f"estimated mean density ({nbar:g}) by more than 3 sigma.",
                )
            nbar = nbar_

        # normalize map
        pos /= nbar

        # compute density contrast if asked to
        if self.__overdensity:
            if vmap is None:
                pos -= 1
            else:
                pos -= vmap

        # compute bias of number counts
        bias = ngal / (4 * np.pi) * mapper.area**2 / nbar**2

        # set metadata of array
        update_metadata(pos, catalog, nbar=nbar, bias=bias)

        # return the position map
        return pos


class ScalarField(Field, spin=0):
    """Field of real scalar values in a catalogue."""

    uses = "longitude", "latitude", "value", "[weight]"

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map real values from catalogue to HEALPix map."""

        # get mapper
        mapper = self.mapper_or_error

        # get the column definition of the catalogue
        *col, wcol = self.columns_or_error

        # scalar field map
        val = mapper.create(spin=self.spin)

        # total weighted variance from online algorithm
        ngal = 0
        wmean, var = 0.0, 0.0

        # go through pages in catalogue and map values
        async for page in _pages(catalog, progress):
            if wcol is not None:
                page.delete(page[wcol] == 0)

            if page.size:
                lon, lat, v = page.get(*col)
                w = page.get(wcol) if wcol is not None else None

                mapper.map_values(lon, lat, [val], [v], w)

                ngal += page.size
                if w is None:
                    var += (v**2 - var).sum() / ngal
                else:
                    wmean += (w - wmean).sum() / ngal
                    var += ((w * v) ** 2 - var).sum() / ngal

                del lon, lat, v, w

            # clean up and yield control to main loop
            del page

        # fix mean weight if there was no column for it
        if wcol is None:
            wmean = 1.0

        # compute mean visibility
        if catalog.visibility is None:
            vbar = 1
        else:
            vbar = np.mean(catalog.visibility)

        # compute mean weight per effective mapper "pixel"
        wbar = ngal / (4 * np.pi * vbar) * wmean * mapper.area

        # normalise the map
        val /= wbar

        # compute bias from variance (per object)
        bias = 4 * np.pi * vbar**2 * (var / wmean**2) / ngal

        # set metadata of array
        update_metadata(val, catalog, wbar=wbar, bias=bias)

        # return the value map
        return val


class ComplexField(Field, spin=0):
    """Field of complex values in a catalogue.

    The :class:`ComplexField` class has zero spin weight, while
    subclasses such as :class:`Spin2Field` have non-zero spin weight.

    """

    uses = "longitude", "latitude", "real", "imag", "[weight]"

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map complex values from catalogue to HEALPix map."""

        # get mapper
        mapper = self.mapper_or_error

        # get the column definition of the catalogue
        *col, wcol = self.columns_or_error

        # complex map with real and imaginary part
        val = mapper.create(2, spin=self.spin)

        # total weighted variance from online algorithm
        ngal = 0
        wmean, var = 0.0, 0.0

        # go through pages in catalogue and get the shear values,
        async for page in _pages(catalog, progress):
            if wcol is not None:
                page.delete(page[wcol] == 0)

            if page.size:
                lon, lat, re, im = page.get(*col)

                w = page.get(wcol) if wcol is not None else None

                mapper.map_values(lon, lat, [val[0], val[1]], [re, im], w)

                ngal += page.size
                if w is None:
                    var += (re**2 + im**2 - var).sum() / ngal
                else:
                    wmean += (w - wmean).sum() / ngal
                    var += ((w * re) ** 2 + (w * im) ** 2 - var).sum() / ngal

                del lon, lat, re, im, w

            del page

        # set mean weight if there was no column for it
        if wcol is None:
            wmean = 1.0

        # compute mean visibility
        if catalog.visibility is None:
            vbar = 1
        else:
            vbar = np.mean(catalog.visibility)

        # mean weight per effective mapper "pixel"
        wbar = ngal / (4 * np.pi * vbar) * wmean * mapper.area

        # normalise the map
        val /= wbar

        # bias from measured variance, for E/B decomposition
        bias = 2 * np.pi * vbar**2 * (var / wmean**2) / ngal

        # set metadata of array
        update_metadata(val, catalog, wbar=wbar, bias=bias)

        # return the shear map
        return val


class Visibility(Field, spin=0):
    """Copy visibility map from catalogue at given resolution."""

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Create a visibility map from the given catalogue."""

        # get mapper
        mapper = self.mapper_or_error

        # make sure that catalogue has a visibility map
        vmap = catalog.visibility
        if vmap is None:
            msg = "no visibility map in catalog"
            raise ValueError(msg)

        # create new visibility map
        out = mapper.create(spin=self.spin)

        # warn if visibility is changing resolution
        if vmap.size != out.size:
            import healpy as hp

            warnings.warn(
                f"changing NSIDE of visibility map "
                f"from {hp.get_nside(vmap)} to {mapper.nside}",
            )
            out[:] = hp.ud_grade(vmap, mapper.nside)
        else:
            # copy pixel values
            out[:] = vmap

        update_metadata(out, catalog)

        return out


class Weights(Field, spin=0):
    """Field of weight values from a catalogue."""

    uses = "longitude", "latitude", "[weight]"

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map catalogue weights."""

        # get mapper
        mapper = self.mapper_or_error

        # get the columns for this field
        *col, wcol = self.columns_or_error

        # weight map
        wht = mapper.create(spin=self.spin)

        # total weighted variance from online algorithm
        ngal = 0
        wmean, w2mean = 0.0, 0.0

        # map catalogue
        async for page in _pages(catalog, progress):
            if wcol is not None:
                page.delete(page[wcol] == 0)

            if page.size:
                lon, lat = page.get(*col)

                w = page.get(wcol) if wcol is not None else None

                mapper.map_values(lon, lat, [wht], None, w)

                ngal += page.size
                if w is not None:
                    wmean += (w - wmean).sum() / ngal
                    w2mean += (w**2 - w2mean).sum() / ngal

                del lon, lat, w

            del page

        # set mean weight if there was no column for it
        if wcol is None:
            wmean = w2mean = 1.0

        # compute mean visibility
        if catalog.visibility is None:
            vbar = 1
        else:
            vbar = np.mean(catalog.visibility)

        # mean weight per effective mapper "pixel"
        wbar = ngal / (4 * np.pi * vbar) * wmean * mapper.area

        # normalise the map
        wht /= wbar

        # bias from weights
        bias = 4 * np.pi * vbar**2 * (w2mean / wmean**2) / ngal

        # set metadata of array
        update_metadata(wht, catalog, wbar=wbar, bias=bias)

        # return the weight map
        return wht


class Spin2Field(ComplexField, spin=2):
    """Spin-2 complex field."""


Shears = Spin2Field
Ellipticities = Spin2Field


def get_masks(
    fields: Mapping[str, Field],
    *,
    comb: int | None = None,
    include: Sequence[Sequence[str]] | None = None,
    exclude: Sequence[Sequence[str]] | None = None,
    append_eb: bool = False,
) -> Sequence[str] | Sequence[tuple[str, ...]]:
    """
    Return the masks for a given set of fields.

    If *comb* is given, produce combinations of masks for combinations
    of a number *comb* of fields.

    The fields (not masks) can be filtered using the *include* and
    *exclude* parameters.  If *append_eb* is true, the filter is applied
    to field names including the E/B-mode suffix when the spin weight is
    non-zero.

    """

    isgood = partial(toc_match, include=include, exclude=exclude)

    def _key_eb(key: str) -> tuple[str, ...]:
        """Return the key of the given field with _E/_B appended (or not)."""
        if append_eb and fields[key].spin != 0:
            return (f"{key}_E", f"{key}_B")
        return (key,)

    def _all_str(seq: tuple[str | None, ...]) -> TypeGuard[tuple[str, ...]]:
        """Return true if all items in *seq* are strings."""
        return not any(item is None for item in seq)

    if comb is None:
        masks_no_comb: list[str] = []
        for key, field in fields.items():
            if field.mask is None:
                continue
            if not any(map(isgood, _key_eb(key))):
                continue
            masks_no_comb.append(field.mask)
        return masks_no_comb

    masks_comb: list[tuple[str, ...]] = []
    for keys in combinations_with_replacement(fields, comb):
        item = tuple(fields[key].mask for key in keys)
        if not _all_str(item):
            continue
        if not any(map(isgood, product(*map(_key_eb, keys)))):
            continue
        masks_comb.append(item)
    return masks_comb
