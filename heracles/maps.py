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
"""module for map-making."""

import abc
import collections
import functools
import logging
import typing
import warnings

import healpy as hp
import numba
import numpy as np

from .util import Progress, toc_match

if typing.TYPE_CHECKING:
    from .catalog import Catalog, CatalogPage

logger = logging.getLogger(__name__)


def _nativebyteorder(fn):
    """Utility decorator to convert inputs to native byteorder."""

    @functools.wraps(fn)
    def wrapper(*inputs):
        native = []
        for a in inputs:
            if a.dtype.byteorder != "=":
                a = a.byteswap().newbyteorder("=")
            native.append(a)
        return fn(*native)

    return wrapper


@_nativebyteorder
@numba.njit(nogil=True, fastmath=True)
def _map_pos(pos, ipix):
    for i in ipix:
        pos[i] += 1


@_nativebyteorder
@numba.njit(nogil=True, fastmath=True)
def _map_real(wht, val, ipix, w, v):
    for i, w_i, v_i in zip(ipix, w, v):
        wht[i] += w_i
        val[i] += w_i / wht[i] * (v_i - val[i])


@_nativebyteorder
@numba.njit(nogil=True, fastmath=True)
def _map_complex(wht, val, ipix, w, re, im):
    for i, w_i, re_i, im_i in zip(ipix, w, re, im):
        wht[i] += w_i
        val[0, i] += w_i / wht[i] * (re_i - val[0, i])
        val[1, i] += w_i / wht[i] * (im_i - val[1, i])


@_nativebyteorder
@numba.njit(nogil=True, fastmath=True)
def _map_weight(wht, ipix, w):
    for i, w_i in zip(ipix, w):
        wht[i] += w_i


def update_metadata(array, **metadata):
    """Update metadata of an array dtype."""
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    md.update(metadata)
    # create the new dtype with only the new metadata
    dt = array.dtype
    dt = dt.fields if dt.fields is not None else dt.str
    dt = np.dtype(dt, metadata=md)
    # check that new dtype is compatible with old one
    if not np.can_cast(dt, array.dtype, casting="no"):
        msg = "array with unsupported dtype"
        raise ValueError(msg)
    # set the new dtype in array
    array.dtype = dt


# type alias for map data
MapData = np.ndarray

# type hint for functions returned by map generators
MapFunction = typing.Callable[["CatalogPage"], None]

# type hint for map generators
MapGenerator = typing.Generator[MapFunction, None, MapData]


class Map(metaclass=abc.ABCMeta):
    """Abstract base class for map making from catalogues.

    Concrete classes must implement the `__call__()` method which takes a
    catalogue instance and returns a generator for mapping.

    """

    def __init__(self, columns: tuple[typing.Optional[str]]) -> None:
        """Initialise the map."""
        self._columns = columns
        super().__init__()

    @property
    def columns(self) -> tuple[typing.Optional[str]]:
        """Return the catalogue columns used by this map."""
        return self._columns

    @abc.abstractmethod
    def __call__(self, catalog: "Catalog") -> typing.Union[MapData, MapGenerator]:
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

    def __init__(self, randomize: bool, **kwargs) -> None:
        """Initialise map with the given randomize property."""
        self._randomize = randomize
        super().__init__(**kwargs)

    @property
    def randomize(self) -> bool:
        return self._randomize

    @randomize.setter
    def randomize(self, randomize: bool) -> None:
        """Set the randomize flag."""
        self._randomize = randomize


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
    ) -> None:
        """Create a position map with the given properties."""
        super().__init__(columns=(lon, lat), nside=nside, randomize=randomize)
        self._overdensity: bool = overdensity

    @property
    def overdensity(self) -> bool:
        """Flag to create overdensity maps."""
        return self._overdensity

    @overdensity.setter
    def overdensity(self, overdensity: bool) -> None:
        self._overdensity = overdensity

    def __call__(self, catalog: "Catalog") -> MapGenerator:
        """Map the given catalogue."""
        # get catalogue column definition
        col = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # position map
        pos = np.zeros(npix, dtype=np.float64)

        # keep track of the total number of galaxies
        ngal = 0

        # function to map catalogue data
        def mapper(page: "CatalogPage") -> None:
            nonlocal ngal

            if not self._randomize:
                lon, lat = page.get(*col)
                ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
                _map_pos(pos, ipix)

            ngal += page.size

        # call yields here to apply mapper over entire catalogue
        yield mapper

        # get visibility map if present in catalogue
        vmap = catalog.visibility

        # match resolution of visibility map if present
        if vmap is not None and hp.get_nside(vmap) != self.nside:
            warnings.warn("position and visibility maps have different NSIDE")
            vmap = hp.ud_grade(vmap, self.nside)

        # randomise position map if asked to
        if self._randomize:
            p = np.full(npix, 1 / npix) if vmap is None else vmap / np.sum(vmap)
            pos[:] = np.random.multinomial(ngal, p)

        # compute average number density
        nbar = ngal / npix
        if vmap is not None:
            nbar /= np.mean(vmap)

        # compute overdensity if asked to
        if self._overdensity:
            pos /= nbar
            pos -= 1 if vmap is None else vmap
            power = 0
        else:
            power = 1

        # set metadata of array
        update_metadata(pos, spin=0, nbar=nbar, kernel="healpix", power=power)

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
        weight: typing.Optional[str] = None,
        *,
        normalize: bool = True,
    ) -> None:
        """Create a new real map."""
        super().__init__(
            columns=(lon, lat, value, weight),
            nside=nside,
            normalize=normalize,
        )

    def __call__(self, catalog: "Catalog") -> MapGenerator:
        """Map real values from catalogue to HEALPix map."""
        # get the column definition of the catalogue
        *col, wcol = self.columns

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight and value map
        wht = np.zeros(npix)
        val = np.zeros(npix)

        # go through pages in catalogue and map values
        def mapper(page: "CatalogPage") -> None:
            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, v = page.get(*col)

            w = np.ones(page.size) if wcol is None else page.get(wcol)
            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_real(wht, val, ipix, w, v)

        # call yields here to apply mapper over entire catalogue
        yield mapper

        # compute average weight in nonzero pixels
        wbar = wht.mean()

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
        else:
            power = 1

        # value was averaged in each pixel for numerical stability
        # now compute the sum
        val *= wht

        # set metadata of array
        update_metadata(val, spin=0, wbar=wbar, kernel="healpix", power=power)

        # return the value map
        return val


class ComplexMap(HealpixMap, NormalizableMap, RandomizableMap):
    """Create HEALPix maps from complex values in a catalogue.

    Complex maps can have non-zero spin weight, set using the ``spin=``
    parameter.

    Can optionally flip the sign of the second shear component,
    depending on the ``conjugate`` property.

    """

    def __init__(
        self,
        nside: int,
        lon: str,
        lat: str,
        real: str,
        imag: str,
        weight: typing.Optional[str] = None,
        *,
        spin: int = 0,
        conjugate: bool = False,
        normalize: bool = True,
        randomize: bool = False,
    ) -> None:
        """Create a new shear map."""
        self._spin: int = spin
        self._conjugate: bool = conjugate
        super().__init__(
            columns=(lon, lat, real, imag, weight),
            nside=nside,
            normalize=normalize,
            randomize=randomize,
        )

    @property
    def spin(self) -> int:
        """Spin weight of map."""
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        """Set the spin weight."""
        self._spin = spin

    @property
    def conjugate(self) -> bool:
        """Flag to conjugate shear maps."""
        return self._conjugate

    @conjugate.setter
    def conjugate(self, conjugate: bool) -> None:
        """Set the conjugate flag."""
        self._conjugate = conjugate

    def __call__(self, catalog: "Catalog") -> MapGenerator:
        """Map shears from catalogue to HEALPix map."""
        # get the column definition of the catalogue
        *col, wcol = self.columns

        # get the map properties
        conjugate = self.conjugate
        randomize = self.randomize

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight and shear map
        wht = np.zeros(npix)
        val = np.zeros((2, npix))

        # go through pages in catalogue and get the shear values,
        # randomise if asked to, and do the mapping
        def mapper(page: "CatalogPage") -> None:
            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, re, im = page.get(*col)

            w = np.ones(page.size) if wcol is None else page.get(wcol)
            if conjugate:
                im = -im

            if randomize:
                a = np.random.uniform(0.0, 2 * np.pi, size=page.size)
                r = np.hypot(re, im)
                re, im = r * np.cos(a), r * np.sin(a)
                del a, r

            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_complex(wht, val, ipix, w, re, im)

        # call yields here to apply mapper over entire catalogue
        yield mapper

        # compute average weight in nonzero pixels
        wbar = wht.mean()

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
        else:
            power = 1

        # value was averaged in each pixel for numerical stability
        # now compute the sum
        val *= wht

        # set metadata of array
        update_metadata(val, spin=self.spin, wbar=wbar, kernel="healpix", power=power)

        # return the shear map
        return val


class VisibilityMap(HealpixMap):
    """Copy visibility map from catalogue at given resolution."""

    def __init__(self, nside: int) -> None:
        """Create visibility map at given NSIDE parameter."""
        super().__init__(columns=(), nside=nside)

    def __call__(self, catalog: "Catalog") -> MapData:
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

        update_metadata(vmap, spin=0, kernel="healpix", power=0)

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

    def __call__(self, catalog: "Catalog") -> MapGenerator:
        """Map catalogue weights."""
        # get the columns for this map
        *col, wcol = self.columns

        # number of pixels for nside
        nside = self.nside
        npix = hp.nside2npix(nside)

        # create the weight map
        wht = np.zeros(npix)

        # map catalogue
        def mapper(page: "CatalogPage") -> None:
            lon, lat = page.get(*col)

            w = np.ones(page.size) if wcol is None else page.get(wcol)
            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)

            _map_weight(wht, ipix, w)

        # call yields here to apply mapper over entire catalogue
        yield mapper

        # compute average weight in nonzero pixels
        wbar = wht.mean()

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
        else:
            power = 1

        # set metadata of arrays
        update_metadata(wht, spin=0, wbar=wbar, kernel="healpix", power=power)

        # return the weight map
        return wht


Spin2Map = functools.partial(ComplexMap, spin=2)
ShearMap = Spin2Map
EllipticityMap = Spin2Map


def _close_and_return(generator):
    try:
        next(generator)
    except StopIteration as end:
        return end.value
    else:
        msg = "generator did not stop"
        raise RuntimeError(msg)


def map_catalogs(
    maps: typing.Mapping[typing.Any, Map],
    catalogs: typing.Mapping[typing.Any, "Catalog"],
    *,
    parallel: bool = False,
    out: typing.Optional[typing.MutableMapping[typing.Any, typing.Any]] = None,
    include: typing.Optional[typing.Sequence[tuple[typing.Any, typing.Any]]] = None,
    exclude: typing.Optional[typing.Sequence[tuple[typing.Any, typing.Any]]] = None,
    progress: bool = False,
) -> dict[tuple[typing.Any, typing.Any], MapData]:
    """Make maps for a set of catalogues."""
    # the toc dict of maps
    if out is None:
        out = {}

    # display a progress bar if asked to
    if progress:
        prog = Progress()

    # collect groups of catalogues to go through if parallel
    if parallel:
        groups = {}
        for i, catalog in catalogs.items():
            if (catalog.size, catalog.page_size) not in groups:
                groups[catalog.size, catalog.page_size] = {}
            groups[catalog.size, catalog.page_size][i] = catalog
        # unnamed groups of catalogues
        items = ((None, group) for group in groups.values())
    else:
        # consider each catalogue its own named group
        items = ((i, {i: catalog}) for i, catalog in catalogs.items())

    # map each group: run through catalogues and maps
    for name, group in items:
        # apply the maps to each catalogue in the group
        results = {}
        if progress:
            prog.start(len(maps) * len(group), name)
        for i, catalog in group.items():
            for k, mapper in maps.items():
                if toc_match((k, i), include, exclude):
                    results[k, i] = mapper(catalog)
                if progress:
                    prog.update()
        if progress:
            prog.stop()

        # collect map generators from results
        # if there are any generators, feed them the catalogue pages
        if gen := {
            ki: v
            for ki, v in results.items()
            if isinstance(v, collections.abc.Generator)
        }:
            # get an iterator over each catalogue
            # by construction, these have all the same length and page size
            its = {i: iter(catalog) for i, catalog in group.items()}

            # get the iteration limits of this group from first catalog
            size, page_size = next((c.size, c.page_size) for c in group.values())

            # get the mapping functions from each generator
            fns = {ki: next(g) for ki, g in gen.items()}

            # go through each page of each catalogue once
            # give each page to each generator
            # make copies to that generators can delete() etc.
            if progress:
                prog.start(size, name)
            for rownum in range(0, size, page_size):
                for i, it in its.items():
                    try:
                        page = next(it)
                    except StopIteration:
                        msg = (
                            f"catalog {i} finished prematurely in "
                            f"page started at row {rownum}"
                        )
                        raise RuntimeError(
                            msg,
                        )
                    for k in maps:
                        if (fn := fns.get((k, i))) is not None:
                            fn(page.copy())
                if progress:
                    prog.update(page_size)
            if progress:
                prog.stop()

            # done with the mapping functions
            del fn, fns

            # terminate generators and store results
            if progress:
                prog.start(len(gen), name)
            for ki, g in gen.items():
                results[ki] = _close_and_return(g)
                if progress:
                    prog.update()
            if progress:
                prog.stop()

            # done with the generators
            del gen

        # store results
        for ki, v in results.items():
            out[ki] = v

        # results are no longer needed
        del results

    # return the toc dict
    return out


def transform_maps(
    maps: typing.Mapping[tuple[typing.Any, typing.Any], MapData],
    *,
    out: typing.Optional[typing.MutableMapping[typing.Any, typing.Any]] = None,
    progress: bool = False,
    **kwargs,
) -> dict[tuple[typing.Any, typing.Any], np.ndarray]:
    """Transform a set of maps to alms."""
    # the output toc dict
    if out is None:
        out = {}

    # display a progress bar if asked to
    if progress:
        prog = Progress()
        prog.start(len(maps))

    # convert maps to alms, taking care of complex and spin-weighted maps
    for (k, i), m in maps.items():
        nside = hp.get_nside(m)

        md = m.dtype.metadata or {}
        spin = md.get("spin", 0)

        logger.info(f"transforming {k} map (spin {spin}) for bin {i}")

        if spin == 0:
            pol = False
        elif spin == 2:
            pol = True
            m = [np.zeros(np.shape(m)[-1]), m[0], m[1]]
        else:
            msg = f"spin-{spin} maps not yet supported"
            raise NotImplementedError(msg)

        alms = hp.map2alm(m, pol=pol, **kwargs)

        if spin == 0:
            alms = {(k, i): alms}
        elif spin == 2:
            alms = {(f"{k}_E", i): alms[1], (f"{k}_B", i): alms[2]}

        for ki, alm in alms.items():
            if md:
                update_metadata(alm, nside=nside, **md)
            out[ki] = alm

        del m, alms, alm

        if progress:
            prog.update()

    if progress:
        prog.stop()

    # return the toc dict of alms
    return out
