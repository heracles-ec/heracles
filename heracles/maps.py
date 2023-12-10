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
"""module for map-making"""

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from functools import partial, wraps
from typing import TYPE_CHECKING

import coroutines
import healpy as hp
import numpy as np
from numba import njit

from .core import TocDict, toc_match

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, MutableMapping, Sequence
    from typing import Any

    from numpy.typing import ArrayLike

    from .catalog import Catalog, CatalogPage
    from .progress import Progress, ProgressTask


def _nativebyteorder(fn):
    """utility decorator to convert inputs to native byteorder"""

    @wraps(fn)
    def wrapper(*inputs):
        native = []
        for a in inputs:
            if a.dtype.byteorder != "=":
                a = a.byteswap().newbyteorder("=")
            native.append(a)
        return fn(*native)

    return wrapper


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_pos(pos, ipix):
    for i in ipix:
        pos[i] += 1


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_real(wht, val, ipix, w, v):
    for i, w_i, v_i in zip(ipix, w, v):
        wht[i] += w_i
        val[i] += w_i / wht[i] * (v_i - val[i])


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_complex(wht, val, ipix, w, re, im):
    for i, w_i, re_i, im_i in zip(ipix, w, re, im):
        wht[i] += w_i
        val[0, i] += w_i / wht[i] * (re_i - val[0, i])
        val[1, i] += w_i / wht[i] * (im_i - val[1, i])


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_weight(wht, ipix, w):
    for i, w_i in zip(ipix, w):
        wht[i] += w_i


def update_metadata(array, **metadata):
    """update metadata of an array dtype"""
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    md.update(metadata)
    # create the new dtype with only the new metadata
    dt = array.dtype
    if dt.fields is not None:
        dt = dt.fields
    else:
        dt = dt.str
    dt = np.dtype(dt, metadata=md)
    # check that new dtype is compatible with old one
    if not np.can_cast(dt, array.dtype, casting="no"):
        msg = "array with unsupported dtype"
        raise ValueError(msg)
    # set the new dtype in array
    array.dtype = dt


class Map(metaclass=ABCMeta):
    """Abstract base class for map making from catalogues.

    Concrete classes must implement the `__call__()` method which takes a
    catalogue instance and returns a generator for mapping.

    """

    def __init__(self, columns: tuple[str | None]) -> None:
        """Initialise the map."""
        self._columns = columns
        super().__init__()

    @property
    def columns(self) -> tuple[str | None]:
        """Return the catalogue columns used by this map."""
        return self._columns

    @abstractmethod
    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Implementation for mapping a catalogue."""
        ...


class HealpixMap(Map):
    """Abstract base class for HEALPix map making.

    HEALPix maps have a resolution parameter, available as the ``nside``
    property.

    """

    def __init__(self, nside: int, **kwargs) -> None:
        """Initialize map with the given nside parameter."""
        self._nside: int = nside
        super().__init__(**kwargs)

    @property
    def nside(self) -> int:
        """The resolution parameter of the HEALPix map."""
        return self._nside

    @nside.setter
    def nside(self, nside: int) -> None:
        """Set the resolution parameter of the HEALPix map."""
        self._nside = nside


class RandomizableMap(Map):
    """Abstract base class for randomisable maps.

    Randomisable maps have a ``randomize`` property that determines
    whether or not the maps are randomised.

    """

    default_rng: np.random.Generator = np.random.default_rng()
    """Default random number generator for randomizable maps."""

    def __init__(
        self,
        randomize: bool,
        *,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> None:
        """Initialise map with the given randomize property."""
        self._randomize = randomize
        self._rng = rng
        super().__init__(**kwargs)

    @property
    def randomize(self) -> bool:
        return self._randomize

    @randomize.setter
    def randomize(self, randomize: bool) -> None:
        """Set the randomize flag."""
        self._randomize = randomize

    @property
    def rng(self) -> np.random.Generator:
        """Random number generator of this map."""
        return self._rng or self.default_rng

    @rng.setter
    def rng(self, rng: np.random.Generator) -> None:
        """Set the random number generator of this map."""
        self._rng = rng


class NormalizableMap(Map):
    """Abstract base class for normalisable maps.

    A normalised map is a map that is divided by its mean weight.

    Normalisable maps have a ``normalize`` property that determines
    whether or not the maps are normalised.

    """

    def __init__(self, normalize: bool, **kwargs) -> None:
        """Initialise map with the given normalize property."""
        self._normalize = normalize
        super().__init__(**kwargs)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        """Set the normalize flag."""
        self._normalize = normalize


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


class PositionMap(HealpixMap, RandomizableMap):
    """Create HEALPix maps from positions in a catalogue.

    Can produce both overdensity maps and number count maps, depending
    on the ``overdensity`` property.

    """

    def __init__(
        self,
        nside: int,
        lon: str,
        lat: str,
        *,
        overdensity: bool = True,
        randomize: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Create a position map with the given properties."""
        super().__init__(columns=(lon, lat), nside=nside, randomize=randomize, rng=rng)
        self._overdensity: bool = overdensity

    @property
    def overdensity(self) -> bool:
        """Flag to create overdensity maps."""
        return self._overdensity

    @overdensity.setter
    def overdensity(self, overdensity: bool) -> None:
        self._overdensity = overdensity

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map the given catalogue."""

        # get catalogue column definition
        col = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # position map
        pos = np.zeros(npix, dtype=np.float64)

        # keep track of the total number of galaxies
        ngal = 0

        # map catalogue data asynchronously
        async for page in _pages(catalog, progress):
            if not self._randomize:
                lon, lat = page.get(*col)
                ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
                _map_pos(pos, ipix)

            ngal += page.size

            # clean up to free unneeded memory
            del page, lon, lat

        # get visibility map if present in catalogue
        vmap = catalog.visibility

        # match resolution of visibility map if present
        if vmap is not None and hp.get_nside(vmap) != self.nside:
            warnings.warn("position and visibility maps have different NSIDE")
            vmap = hp.ud_grade(vmap, self.nside)

        # randomise position map if asked to
        if self._randomize:
            if vmap is None:
                p = np.full(npix, 1 / npix)
            else:
                p = vmap / np.sum(vmap)
            pos[:] = self.rng.multinomial(ngal, p)

        # compute average number density
        nbar = ngal / npix
        if vmap is None:
            vbar = 1
        else:
            vbar = np.mean(vmap)
            nbar /= vbar

        # compute overdensity if asked to
        if self._overdensity:
            pos /= nbar
            if vmap is None:
                pos -= 1
            else:
                pos -= vmap
            power = 0
            bias = 4 * np.pi * vbar**2 / ngal
        else:
            power = 1
            bias = (4 * np.pi / npix) * (ngal / npix)

        # set metadata of array
        update_metadata(
            pos,
            catalog=catalog.label,
            spin=0,
            nbar=nbar,
            kernel="healpix",
            nside=self.nside,
            power=power,
            bias=bias,
        )

        # return the position map
        return pos


class ScalarMap(HealpixMap, NormalizableMap):
    """Create HEALPix maps from real scalar values in a catalogue."""

    def __init__(
        self,
        nside: int,
        lon: str,
        lat: str,
        value: str,
        weight: str | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        """Create a new real map."""

        super().__init__(
            columns=(lon, lat, value, weight),
            nside=nside,
            normalize=normalize,
        )

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map real values from catalogue to HEALPix map."""

        # get the column definition of the catalogue
        *col, wcol = self.columns

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight and value map
        wht = np.zeros(npix)
        val = np.zeros(npix)

        # total weighted variance from online algorithm
        ngal = 0
        wmean, var = 0.0, 0.0

        # go through pages in catalogue and map values
        async for page in _pages(catalog, progress):
            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, v = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_real(wht, val, ipix, w, v)

            if page.size:
                ngal += page.size
                wmean += (w - wmean).sum() / ngal
                var += ((w * v) ** 2 - var).sum() / ngal

            # clean up and yield control to main loop
            del page, lon, lat, v, w

        # compute mean visibility
        if catalog.visibility is None:
            vbar = 1
        else:
            vbar = np.mean(catalog.visibility)

        # compute mean weight per visible pixel
        wbar = ngal / npix / vbar * wmean

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
            bias = 4 * np.pi * vbar**2 / ngal * (var / wmean**2)
        else:
            power = 1
            bias = (4 * np.pi / npix) * (ngal / npix) * var

        # value was averaged in each pixel for numerical stability
        # now compute the sum
        val *= wht

        # set metadata of array
        update_metadata(
            val,
            catalog=catalog.label,
            spin=0,
            wbar=wbar,
            kernel="healpix",
            nside=self.nside,
            power=power,
            bias=bias,
        )

        # return the value map
        return val


class ComplexMap(HealpixMap, NormalizableMap, RandomizableMap):
    """Create HEALPix maps from complex values in a catalogue.

    Complex maps can have non-zero spin weight, set using the ``spin=``
    parameter.

    """

    def __init__(
        self,
        nside: int,
        lon: str,
        lat: str,
        real: str,
        imag: str,
        weight: str | None = None,
        *,
        spin: int = 0,
        normalize: bool = True,
        randomize: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Create a new shear map."""

        self._spin: int = spin
        super().__init__(
            columns=(lon, lat, real, imag, weight),
            nside=nside,
            normalize=normalize,
            randomize=randomize,
            rng=rng,
        )

    @property
    def spin(self) -> int:
        """Spin weight of map."""
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        """Set the spin weight."""
        self._spin = spin

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map shears from catalogue to HEALPix map."""

        # get the column definition of the catalogue
        *col, wcol = self.columns

        # get the map properties
        randomize = self.randomize

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight and shear map
        wht = np.zeros(npix)
        val = np.zeros((2, npix))

        # total weighted variance from online algorithm
        ngal = 0
        wmean, var = 0.0, 0.0

        # go through pages in catalogue and get the shear values,
        # randomise if asked to, and do the mapping
        async for page in _pages(catalog, progress):
            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, re, im = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            if randomize:
                a = self.rng.uniform(0.0, 2 * np.pi, size=page.size)
                r = np.hypot(re, im)
                re, im = r * np.cos(a), r * np.sin(a)
                del a, r

            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_complex(wht, val, ipix, w, re, im)

            if page.size:
                ngal += page.size
                wmean += (w - wmean).sum() / ngal
                var += ((w * re) ** 2 + (w * im) ** 2 - var).sum() / ngal

            del page, lon, lat, re, im, w

        # compute mean visibility
        if catalog.visibility is None:
            vbar = 1
        else:
            vbar = np.mean(catalog.visibility)

        # mean weight per visible pixel
        wbar = ngal / npix / vbar * wmean

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
            bias = 2 * np.pi * vbar**2 / ngal * (var / wmean**2)
        else:
            power = 1
            bias = (2 * np.pi / npix) * (ngal / npix) * var

        # value was averaged in each pixel for numerical stability
        # now compute the sum
        val *= wht

        # set metadata of array
        update_metadata(
            val,
            catalog=catalog.label,
            spin=self.spin,
            wbar=wbar,
            kernel="healpix",
            nside=self.nside,
            power=power,
            bias=bias,
        )

        # return the shear map
        return val


class VisibilityMap(HealpixMap):
    """Copy visibility map from catalogue at given resolution."""

    def __init__(self, nside: int) -> None:
        """Create visibility map at given NSIDE parameter."""
        super().__init__(columns=(), nside=nside)

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Create a visibility map from the given catalogue."""

        # make sure that catalogue has a visibility map
        vmap = catalog.visibility
        if vmap is None:
            msg = "no visibility map in catalog"
            raise ValueError(msg)

        # warn if visibility is changing resolution
        vmap_nside = hp.get_nside(vmap)
        if vmap_nside != self.nside:
            warnings.warn(
                f"changing NSIDE of visibility map "
                f"from {vmap_nside} to {self.nside}",
            )
            vmap = hp.ud_grade(vmap, self.nside)
        else:
            # make a copy for updates to metadata
            vmap = np.copy(vmap)

        update_metadata(
            vmap,
            catalog=catalog.label,
            spin=0,
            kernel="healpix",
            nside=self.nside,
            power=0,
        )

        return vmap


class WeightMap(HealpixMap, NormalizableMap):
    """Create a HEALPix weight map from a catalogue."""

    def __init__(
        self,
        nside: int,
        lon: str,
        lat: str,
        weight: str,
        *,
        normalize=True,
    ) -> None:
        """Create a new weight map."""
        super().__init__(columns=(lon, lat, weight), nside=nside, normalize=normalize)

    async def __call__(
        self,
        catalog: Catalog,
        *,
        progress: ProgressTask | None = None,
    ) -> ArrayLike:
        """Map catalogue weights."""

        # get the columns for this map
        *col, wcol = self.columns

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight map
        wht = np.zeros(npix)

        # map catalogue
        async for page in _pages(catalog, progress):
            lon, lat = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_weight(wht, ipix, w)

            del page, lon, lat, w

        # compute average weight in nonzero pixels
        wbar = wht.mean()
        if catalog.visibility is not None:
            wbar /= np.mean(catalog.visibility)

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
        else:
            power = 1

        # set metadata of arrays
        update_metadata(
            wht,
            catalog=catalog.label,
            spin=0,
            wbar=wbar,
            kernel="healpix",
            nside=self.nside,
            power=power,
        )

        # return the weight map
        return wht


Spin2Map = partial(ComplexMap, spin=2)
ShearMap = Spin2Map
EllipticityMap = Spin2Map


async def _mapper_task(
    key: tuple[Any, ...],
    mapper: Map,
    catalog: Catalog,
    progress: Progress,
) -> ArrayLike:
    """
    Removes the task when coroutine finishes.
    """
    name = "[" + ", ".join(map(str, key)) + "]"
    task = progress.task(name, subtask=True, total=None)
    try:
        return await mapper(catalog, progress=task)
    finally:
        task.remove()
        progress.advance(progress.task_ids[0])


def map_catalogs(
    maps: Mapping[Any, Map],
    catalogs: Mapping[Any, Catalog],
    *,
    parallel: bool = False,
    out: MutableMapping[Any, Any] = None,
    include: Sequence[tuple[Any, Any]] | None = None,
    exclude: Sequence[tuple[Any, Any]] | None = None,
    progress: bool = False,
) -> dict[tuple[Any, Any], ArrayLike]:
    """Make maps for a set of catalogues."""

    # the toc dict of maps
    if out is None:
        out = TocDict()

    # collect groups of items to go through
    # items are tuples of (key, mapper, catalog)
    groups = [
        [((i, j), mapper, catalog) for i, mapper in maps.items()]
        for j, catalog in catalogs.items()
    ]

    # flatten groups for parallel processing
    if parallel:
        groups = [sum(groups, [])]

    # display a progress bar if asked to
    if progress:
        from .progress import Progress

        # create the progress bar
        # add the main task -- this must be the first task
        _progress = Progress()
        _progress.add_task("mapping", total=sum(map(len, groups)))
        _progress.start()

    # process all groups of mappers and catalogues
    for items in groups:
        # mappers return coroutines, which are ran concurrently
        keys, coros = [], []
        for key, mapper, catalog in items:
            if toc_match(key, include, exclude):
                if progress:
                    coro = _mapper_task(key, mapper, catalog, _progress)
                else:
                    coro = mapper(catalog)
                keys.append(key)
                coros.append(coro)

        # run all coroutines concurrently
        results = coroutines.run(coroutines.gather(*coros))

        # store results
        for key, value in zip(keys, results):
            out[key] = value

        # free up memory for next group
        del results

    if progress:
        _progress.refresh()
        _progress.stop()

    # return the toc dict
    return out


def transform_maps(
    maps: Mapping[tuple[Any, Any], ArrayLike],
    *,
    lmax: int | Mapping[Any, int] | None = None,
    out: MutableMapping[Any, Any] = None,
    progress: bool = False,
    **kwargs,
) -> dict[tuple[Any, Any], ArrayLike]:
    """transform a set of maps to alms"""

    # the output toc dict
    if out is None:
        out = TocDict()

    # display a progress bar if asked to
    if progress:
        from .progress import Progress

        _progress = Progress()
        _task = _progress.task("transform", total=len(maps))
        _progress.start()

    # convert maps to alms, taking care of complex and spin-weighted maps
    for (k, i), m in maps.items():
        if isinstance(lmax, Mapping):
            _lmax = lmax.get((k, i)) or lmax.get(k)
        else:
            _lmax = lmax

        md = m.dtype.metadata or {}
        spin = md.get("spin", 0)

        if progress:
            _subtask = _progress.task(
                f"[{k}, {i}]",
                subtask=True,
                start=False,
                total=None,
            )

        if spin == 0:
            pol = False
        elif spin == 2:
            pol = True
            m = [np.zeros(np.shape(m)[-1]), m[0], m[1]]
        else:
            msg = f"spin-{spin} maps not yet supported"
            raise NotImplementedError(msg)

        alms = hp.map2alm(m, lmax=_lmax, pol=pol, **kwargs)

        if spin == 0:
            alms = {(k, i): alms}
        elif spin == 2:
            alms = {(f"{k}_E", i): alms[1], (f"{k}_B", i): alms[2]}

        for ki, alm in alms.items():
            update_metadata(alm, **md)
            out[ki] = alm

        del m, alms, alm

        if progress:
            _subtask.remove()
            _task.update(advance=1)

    if progress:
        _progress.refresh()
        _progress.stop()

    # return the toc dict of alms
    return out
