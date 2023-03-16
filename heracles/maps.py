'''module for map-making'''

import warnings
import time
from datetime import timedelta
from abc import ABCMeta, abstractmethod
from collections.abc import Generator, Sequence, Mapping
from functools import wraps, partial
import logging
import numpy as np
import healpy as hp
from numba import njit

import typing as t
if t.TYPE_CHECKING:
    from .catalog import Catalog, CatalogPage

logger = logging.getLogger(__name__)


def _nativebyteorder(fn):
    '''utility decorator to convert inputs to native byteorder'''

    @wraps(fn)
    def wrapper(*inputs):
        native = []
        for a in inputs:
            if a.dtype.byteorder != '=':
                a = a.byteswap().newbyteorder('=')
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
        val[i] += w_i/wht[i]*(v_i - val[i])


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_complex(wht, val, ipix, w, re, im):
    for i, w_i, re_i, im_i in zip(ipix, w, re, im):
        wht[i] += w_i
        val[0, i] += w_i/wht[i]*(re_i - val[0, i])
        val[1, i] += w_i/wht[i]*(im_i - val[1, i])


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map_weight(wht, ipix, w):
    for i, w_i in zip(ipix, w):
        wht[i] += w_i


def update_metadata(array, **metadata):
    '''update metadata of an array dtype'''
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    md.update(metadata)
    array.dtype = np.dtype(array.dtype, metadata=md)


# type alias for map data
MapData = np.ndarray

# type hint for map generators
MapGenerator = t.Generator[None, 'CatalogPage', MapData]


class Map(metaclass=ABCMeta):
    '''Abstract base class for map making from catalogues.

    Concrete classes must implement the `__call__()` method which takes a
    catalogue instance and returns a generator for mapping.

    '''

    def __init__(self, columns: t.Tuple[t.Optional[str]]) -> None:
        '''Initialise the map.'''
        self._columns = columns
        super().__init__()

    @property
    def columns(self) -> t.Tuple[t.Optional[str]]:
        '''Return the catalogue columns used by this map.'''
        return self._columns

    @abstractmethod
    def __call__(self, catalog: 'Catalog') -> t.Union[MapData, MapGenerator]:
        '''Implementation for mapping a catalogue.'''
        ...


class HealpixMap(Map):
    '''Abstract base class for HEALPix map making.

    HEALPix maps have a resolution parameter, available as the ``nside``
    property.

    '''

    def __init__(self, nside: int, **kwargs) -> None:
        '''Initialize map with the given nside parameter.'''
        self._nside: int = nside
        super().__init__(**kwargs)

    @property
    def nside(self) -> int:
        '''The resolution parameter of the HEALPix map.'''
        return self._nside

    @nside.setter
    def nside(self, nside: int) -> None:
        '''Set the resolution parameter of the HEALPix map.'''
        self._nside = nside


class RandomizableMap(Map):
    '''Abstract base class for randomisable maps.

    Randomisable maps have a ``randomize`` property that determines
    whether or not the maps are randomised.

    '''

    def __init__(self, randomize: bool, **kwargs) -> None:
        '''Initialise map with the given randomize property.'''
        self._randomize = randomize
        super().__init__(**kwargs)

    @property
    def randomize(self) -> bool:
        return self._randomize

    @randomize.setter
    def randomize(self, randomize: bool) -> None:
        '''Set the randomize flag.'''
        self._randomize = randomize


class NormalizableMap(Map):
    '''Abstract base class for normalisable maps.

    A normalised map is a map that is divided by its mean weight.

    Normalisable maps have a ``normalize`` property that determines
    whether or not the maps are normalised.

    '''

    def __init__(self, normalize: bool, **kwargs) -> None:
        '''Initialise map with the given normalize property.'''
        self._normalize = normalize
        super().__init__(**kwargs)

    @property
    def normalize(self) -> bool:
        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        '''Set the normalize flag.'''
        self._normalize = normalize


class PositionMap(HealpixMap, RandomizableMap):
    '''Create HEALPix maps from positions in a catalogue.

    Can produce both overdensity maps and number count maps, depending
    on the ``overdensity`` property.

    '''

    def __init__(self, nside: int, lon: str, lat: str, *,
                 overdensity: bool = True, randomize: bool = False
                 ) -> None:
        '''Create a position map with the given properties.'''
        super().__init__(columns=(lon, lat), nside=nside, randomize=randomize)
        self._overdensity: bool = overdensity

    @property
    def overdensity(self) -> bool:
        '''Flag to create overdensity maps.'''
        return self._overdensity

    @overdensity.setter
    def overdensity(self, overdensity: bool) -> None:
        self._overdensity = overdensity

    def __call__(self, catalog: 'Catalog') -> MapGenerator:
        '''Map the given catalogue.'''

        # get catalogue column definition
        col = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # position map
        pos = np.zeros(npix, dtype=np.float64)

        # keep track of the total number of galaxies
        ngal = 0

        # catalogue pages to map
        while True:
            try:
                page = yield
            except GeneratorExit:
                break

            if not self._randomize:
                lon, lat = page.get(*col)
                ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
                _map_pos(pos, ipix)
                del lon, lat

            ngal += page.size

        # get visibility map if present in catalogue
        vmap = catalog.visibility

        # match resolution of visibility map if present
        if vmap is not None and hp.get_nside(vmap) != self.nside:
            warnings.warn('position and visibility maps have different NSIDE')
            vmap = hp.ud_grade(vmap, self.nside)

        # randomise position map if asked to
        if self._randomize:
            if vmap is None:
                p = np.full(npix, 1/npix)
            else:
                p = vmap/np.sum(vmap)
            pos[:] = np.random.multinomial(ngal, p)

        # compute average number density
        nbar = ngal/npix
        if vmap is not None:
            nbar /= np.mean(vmap)

        # compute overdensity if asked to
        if self._overdensity:
            pos /= nbar
            if vmap is None:
                pos -= 1
            else:
                pos -= vmap
            power = 0
        else:
            power = 1

        # set metadata of array
        update_metadata(pos, spin=0, nbar=nbar, kernel='healpix', power=power)

        # return the position map
        return pos


class RealMap(HealpixMap, NormalizableMap):
    '''Create HEALPix maps from real values in a catalogue.'''

    def __init__(self, nside: int, lon: str, lat: str, value: str,
                 weight: t.Optional[str] = None, *, normalize: bool = True
                 ) -> None:
        '''Create a new real map.'''

        super().__init__(columns=(lon, lat, value, weight), nside=nside,
                         normalize=normalize)

    def __call__(self, catalog: 'Catalog') -> MapGenerator:
        '''Map real values from catalogue to HEALPix map.'''

        # get the column definition of the catalogue
        *col, wcol = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # create the weight and value map
        wht = np.zeros(npix)
        val = np.zeros(npix)

        # go through pages in catalogue and map values
        while True:
            try:
                page = yield
            except GeneratorExit:
                break

            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, v = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)

            _map_real(wht, val, ipix, w, v)

            del lon, lat, v, w

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
        update_metadata(val, spin=0, wbar=wbar, kernel='healpix', power=power)

        # return the value map
        return val


class ComplexMap(HealpixMap, NormalizableMap, RandomizableMap):
    '''Create HEALPix maps from complex values in a catalogue.

    Complex maps can have non-zero spin weight, set using the ``spin=``
    parameter.

    Can optionally flip the sign of the second shear component,
    depending on the ``conjugate`` property.

    '''

    def __init__(self, nside: int, lon: str, lat: str, real: str, imag: str,
                 weight: t.Optional[str] = None, *, spin: int = 0,
                 conjugate: bool = False, normalize: bool = True,
                 randomize: bool = False
                 ) -> None:
        '''Create a new shear map.'''

        self._spin: int = spin
        self._conjugate: bool = conjugate
        super().__init__(columns=(lon, lat, real, imag, weight), nside=nside,
                         normalize=normalize, randomize=randomize)

    @property
    def spin(self) -> int:
        '''Spin weight of map.'''
        return self._spin

    @spin.setter
    def spin(self, spin: int) -> None:
        '''Set the spin weight.'''
        self._spin = spin

    @property
    def conjugate(self) -> bool:
        '''Flag to conjugate shear maps.'''
        return self._conjugate

    @conjugate.setter
    def conjugate(self, conjugate: bool) -> None:
        '''Set the conjugate flag.'''
        self._conjugate = conjugate

    def __call__(self, catalog: 'Catalog') -> MapGenerator:
        '''Map shears from catalogue to HEALPix map.'''

        # get the column definition of the catalogue
        *col, wcol = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # create the weight and shear map
        wht = np.zeros(npix)
        val = np.zeros((2, npix))

        # go through pages in catalogue and get the shear values,
        # randomise if asked to, and do the mapping
        while True:
            try:
                page = yield
            except GeneratorExit:
                break

            if wcol is not None:
                page.delete(page[wcol] == 0)

            lon, lat, re, im = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            if self._conjugate:
                im = -im

            if self.randomize:
                a = np.random.uniform(0., 2*np.pi, size=page.size)
                r = np.hypot(re, im)
                re, im = r*np.cos(a), r*np.sin(a)
                del a, r

            ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)

            _map_complex(wht, val, ipix, w, re, im)

            del lon, lat, re, im, w

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
        update_metadata(val, spin=self.spin, wbar=wbar, kernel='healpix',
                        power=power)

        # return the shear map
        return val


class VisibilityMap(HealpixMap):
    '''Copy visibility map from catalogue at given resolution.'''

    def __init__(self, nside: int) -> None:
        '''Create visibility map at given NSIDE parameter.'''
        super().__init__(columns=(), nside=nside)

    def __call__(self, catalog: 'Catalog') -> MapData:
        '''Create a visibility map from the given catalogue.'''

        # make sure that catalogue has a visibility map
        vmap = catalog.visibility
        if vmap is None:
            raise ValueError('no visibility map in catalog')

        # warn if visibility is changing resolution
        vmap_nside = hp.get_nside(vmap)
        if vmap_nside != self.nside:
            warnings.warn(f'changing NSIDE of visibility map '
                          f'from {vmap_nside} to {self.nside}')
            vmap = hp.ud_grade(vmap, self.nside)
        else:
            # make a copy for updates to metadata
            vmap = np.copy(vmap)

        update_metadata(vmap, spin=0, kernel='healpix', power=0)

        return vmap


class WeightMap(HealpixMap, NormalizableMap):
    '''Create a HEALPix weight map from a catalogue.'''

    def __init__(self, nside: int, lon: str, lat: str, weight: str, *,
                 normalize=True) -> None:
        '''Create a new weight map.'''
        super().__init__(columns=(lon, lat, weight), nside=nside,
                         normalize=normalize)

    def __call__(self, catalog: 'Catalog') -> MapGenerator:
        '''Map catalogue weights.'''

        # get the columns for this map
        *col, wcol = self.columns

        # number of pixels for nside
        npix = hp.nside2npix(self.nside)

        # create the weight map
        wht = np.zeros(npix)

        # map catalogue
        while True:
            try:
                page = yield
            except GeneratorExit:
                break

            lon, lat = page.get(*col)

            if wcol is None:
                w = np.ones(page.size)
            else:
                w = page.get(wcol)

            ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)

            _map_weight(wht, ipix, w)

            del lon, lat, w

        # compute average weight in nonzero pixels
        wbar = wht.mean()

        # normalise the weight in each pixel if asked to
        if self.normalize:
            wht /= wbar
            power = 0
        else:
            power = 1

        # set metadata of arrays
        update_metadata(wht, spin=0, wbar=wbar, kernel='healpix', power=power)

        # return the weight map
        return wht


Spin2Map = partial(ComplexMap, spin=2)
ShearMap = Spin2Map
EllipticityMap = Spin2Map


def _items(obj):
    '''Create an iterator over items for mapping, sequence, or object.'''

    # always convert key to tuple for concatenation
    if isinstance(obj, Mapping):
        return (((i,), v) for i, v in obj.items())
    elif isinstance(obj, Sequence):
        return (((i,), v) for i, v in enumerate(obj))
    else:
        # single catalogue: items are empty tuple + catalogue
        return (((), v) for v in [obj])


def map_catalogs(maps: t.Dict[t.Any, Map],
                 catalogs: t.Dict[t.Any, 'Catalog'],
                 ) -> t.Union[MapData, t.Dict[t.Tuple[t.Any, ...], MapData]]:
    '''Make maps for a set of catalogues.

    The output is a single map, if both ``maps`` and ``catalogs`` are single
    objects, or a dict where the keys are the broadcast of ``maps`` and
    ``catalogs``.

    '''

    t = time.monotonic()

    # the toc dict of maps
    m = {}

    # for computation, go through catalogues first and maps second
    for i, catalog in _items(catalogs):

        logger.info('mapping catalog %s', '' if i == () else i[0])
        ti = time.monotonic()

        # apply the maps to the catalogue
        results = {k: v(catalog) for k, v in _items(maps)}

        # collect generators from results
        gen = {k: v for k, v in results.items() if isinstance(v, Generator)}

        # if there are any generators, feed them the catalogue pages
        if gen:

            # prime the generators for mapping
            for g in gen.values():
                g.send(None)

            # go through catalogue pages once
            # give each page to each generator
            # make copies to that generators can delete() etc.
            for page in catalog:
                for g in gen.values():
                    g.send(page.copy())

            # close generators and store results
            for k, g in gen.items():
                try:
                    g.throw(GeneratorExit)
                except StopIteration as e:
                    results[k] = e.value
                else:
                    results[k] = None

        # store results
        for k, v in results.items():
            j = k + i
            if len(j) == 1:
                m[j[0]] = v
            else:
                m[j] = v

        # results are no longer needed
        del results

        logger.info('mapped catalog %s in %s', '' if i == () else i[0],
                    timedelta(seconds=(time.monotonic() - ti)))

    logger.info('created %d map(s) in %s', len(m),
                timedelta(seconds=(time.monotonic() - t)))

    # return single item if inputs were single item
    if len(m) == 1:
        try:
            return m[()]
        except KeyError:
            pass

    # return maps as a toc dict
    return m


def transform_maps(maps: t.Dict[t.Tuple[t.Any, t.Any], MapData],
                   names: t.Dict[t.Any, t.Any] = {},
                   **kwargs
                   ) -> t.Dict[t.Tuple[t.Any, t.Any], np.ndarray]:
    '''transform a set of maps to alms'''

    logger.info('transforming %d map(s) to alms', len(maps))
    t = time.monotonic()

    # convert maps to alms, taking care of complex and spin-weighted maps
    alms = {}
    for (k, i), m in maps.items():

        nside = hp.get_nside(m)

        md = m.dtype.metadata or {}
        spin = md.get('spin', 0)

        logger.info('transforming %s map (spin %s) for bin %s', k, spin, i)

        if spin == 0:
            pol = False
        elif spin == 2:
            pol = True
            m = [np.zeros(np.shape(m)[-1]), m[0], m[1]]
        else:
            raise NotImplementedError(f'spin-{spin} maps not yet supported')

        a = hp.map2alm(m, pol=pol, **kwargs)

        if spin == 0:
            j = names.get(k, k)
            out = {(j, i): a}
        elif spin == 2:
            j1, j2 = names.get(k, ('E', 'B'))
            out = {(j1, i): a[1], (j2, i): a[2]}

        for j, aj in out.items():
            if j in alms:
                raise KeyError(f'duplicate alm {j}, set `names=` manually')
            if md:
                update_metadata(aj, nside=nside, **md)
            alms[j] = aj

    logger.info('transformed %d map(s) in %s', len(alms), timedelta(seconds=(time.monotonic() - t)))

    # return the toc dict of alms
    return alms
