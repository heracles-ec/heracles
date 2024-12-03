# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
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
"""module for file reading and writing"""

import logging
import os
import re
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Union
from warnings import warn
from weakref import WeakValueDictionary

import fitsio
import numpy as np

from .core import TocDict, toc_match
from .result import Result

if TYPE_CHECKING:
    from typing import TypeAlias

logger = logging.getLogger(__name__)


_METADATA_COMMENTS = {
    "catalog": "catalog of field",
    "catalog_1": "catalog of first field",
    "catalog_2": "catalog of second field",
    "spin": "spin weight of field",
    "spin_1": "spin weight of first field",
    "spin_2": "spin weight of second field",
    "geometry": "mapper geometry of field",
    "geometry_1": "mapper geometry of first field",
    "geometry_2": "mapper geometry of second field",
    "kernel": "mapper kernel of field",
    "kernel_1": "mapper kernel of first field",
    "kernel_2": "mapper kernel of second field",
    "nside": "HEALPix resolution parameter of field",
    "nside_1": "HEALPix resolution parameter of first field",
    "nside_2": "HEALPix resolution parameter of second field",
    "lmax": "LMAX parameter of field",
    "lmax_1": "LMAX parameter of first field",
    "lmax_2": "LMAX parameter of second field",
    "nbar": "mean number count of field",
    "nbar_1": "mean number count of first field",
    "nbar_2": "mean number count of second field",
    "wbar": "mean weight of field",
    "wbar_1": "mean weight of first field",
    "wbar_2": "mean weight of second field",
    "bias": "additive bias of spectrum",
}

# type for valid keys
_DictKey: "TypeAlias" = Union[str, int, tuple["_DictKey", ...]]


def _string_from_key(key: _DictKey) -> str:
    """
    Return string representation for a given key.
    """
    # recursive expansion for sequences
    if isinstance(key, Sequence) and not isinstance(key, str):
        return "-".join(map(_string_from_key, key))

    # get string representation of key
    s = str(key)

    # escape literal "\"
    s = s.replace("\\", "\\\\")

    # escape literal "-"
    s = s.replace("-", "\\-")

    # substitute non-FITS characters by tilde
    s = re.sub(r"[^ -~]+", "~", s, flags=re.ASCII)

    return s


def _key_from_string(s: str) -> _DictKey:
    """
    Return key for a given string representation.
    """
    parts = re.split(r"(?<!\\)-", s.replace("\\\\", "\0"))
    if len(parts) > 1:
        return tuple(map(_key_from_string, parts))
    key = parts[0]
    key = key.replace("\\-", "-")
    key = key.replace("\0", "\\")
    return int(key) if key.removeprefix("-").isdigit() else key


def _get_next_extname(fits, prefix):
    """
    Return the next available extension name starting with *prefix*.
    """
    n = 0
    while (extname := f"{prefix}{n}") in fits:
        n += 1
    return extname


def _iterfits(path, tag, include=None, exclude=None):
    """
    Iterate over HDUs that correspond to *tag* and have valid keys.
    """
    with fitsio.FITS(path) as fits:
        for hdu in fits:
            if not re.match(f"^{tag}\\d+$", hdu.get_extname()):
                continue
            key = _read_key(hdu)
            if key is None:
                continue
            if not toc_match(key, include=include, exclude=exclude):
                continue
            yield key, hdu


def _write_metadata(hdu, metadata):
    """write array metadata to FITS HDU"""
    md = metadata or {}
    for key, value in md.items():
        comment = _METADATA_COMMENTS.get(key, "")
        hdu.write_key("META " + key.upper(), value, comment)


def _read_metadata(hdu):
    """read array metadata from FITS HDU"""
    h = hdu.read_header()
    md = {}
    for key in h:
        if key.startswith("META "):
            md[key[5:].lower()] = h[key]
    return md


def _write_key(hdu, key):
    """write dictionary key to FITS HDU"""
    hdu.write_key("DICTKEY", _string_from_key(key), "dictionary key of this extension")


def _read_key(hdu):
    """read dictionary key from FITS HDU"""
    h = hdu.read_header()
    s = h.get("DICTKEY")
    if s is None:
        return None
    return _key_from_string(s)


def _write_map(fits, ext, key, m, *, names=None):
    """write HEALPix map to FITS table"""

    import healpy as hp

    # prepare column data and names
    cols = list(np.atleast_2d(m))
    if names is None:
        if len(cols) == 1:
            names = ["MAP"]
        else:
            names = [f"MAP{j}" for j in range(1, len(cols) + 1)]

    # write the data
    fits.write_table(cols, names=names, extname=ext)

    # write the key
    _write_key(fits[ext], key)

    # HEALPix metadata
    npix = np.shape(m)[-1]
    nside = hp.npix2nside(npix)
    fits[ext].write_key("PIXTYPE", "HEALPIX", "HEALPIX pixelisation")
    fits[ext].write_key(
        "ORDERING",
        "RING",
        "Pixel ordering scheme, either RING or NESTED",
    )
    fits[ext].write_key("NSIDE", nside, "Resolution parameter of HEALPIX")
    fits[ext].write_key("FIRSTPIX", 0, "First pixel # (0 based)")
    fits[ext].write_key("LASTPIX", npix - 1, "Last pixel # (0 based)")
    fits[ext].write_key(
        "INDXSCHM",
        "IMPLICIT",
        "Indexing: IMPLICIT or EXPLICIT",
    )
    fits[ext].write_key(
        "OBJECT",
        "FULLSKY",
        "Sky coverage, either FULLSKY or PARTIAL",
    )

    # write the metadata
    _write_metadata(fits[ext], m.dtype.metadata)


def _read_map(hdu):
    """read HEALPix map from FITS table"""

    # read the map from the extension
    m = hdu.read()

    # turn the structured array of columns into an unstructured array
    # transpose so that columns become rows (as that is how maps are)
    # then squeeze out degenerate axes
    m = np.squeeze(np.lib.recfunctions.structured_to_unstructured(m).T)

    # read and attach metadata
    m.dtype = np.dtype(m.dtype, metadata=_read_metadata(hdu))

    return m


def _write_complex(fits, ext, key, arr):
    """write complex-valued data to FITS table"""
    # deal with extra dimensions by moving last axis to first
    if arr.ndim > 1:
        arr = np.moveaxis(arr, -1, 0)

    # write the data
    fits.write_table([arr.real, arr.imag], names=["real", "imag"], extname=ext)

    # write the key
    _write_key(fits[ext], key)

    # write the metadata
    _write_metadata(fits[ext], arr.dtype.metadata)


def _read_complex(hdu):
    """read complex-valued data from FITS table"""
    # get column number of real and imag columns
    colnames = hdu.get_colnames()
    col_real, col_imag = colnames.index("real"), colnames.index("imag")
    # get shape of complex array
    _, shape = hdu._get_simple_dtype_and_shape(col_real)
    # create dtype with metadata
    dtype = np.dtype(complex, metadata=_read_metadata(hdu))
    # read complex array from FITS
    arr = np.empty(shape, dtype=dtype)
    arr.real = hdu.read_column(col_real)
    arr.imag = hdu.read_column(col_imag)
    # reorder axes if multidimensional
    if arr.ndim > 1:
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _write_result(fits, ext, key, result):
    """
    Write a result array to FITS.
    """

    # keep ndarray subclasses or we would lose all Result attributes
    result = np.asanyarray(result)

    # get ell axis
    axis = getattr(result, "axis", result.ndim - 1)

    # get data & move ell axis to front
    data = np.moveaxis(result, axis, 0)

    # get ell values or create default
    ell = getattr(result, "ell", None)
    if ell is None:
        ell = np.arange(data.shape[0])

    # get lower bounds or create default
    lower = getattr(result, "lower", None)
    if lower is None:
        lower = ell

    # get upper array bounds or create default
    upper = getattr(result, "upper", None)
    if upper is None:
        upper = np.append(ell[1:], ell[-1] + 1)

    # get weight array or create default
    weight = getattr(result, "weight", None)
    if weight is None:
        weight = np.ones(data.shape[0])

    # write the result as columnar data
    fits.write_table(
        [
            data,
            ell,
            lower,
            upper,
            weight,
        ],
        names=[
            "ARRAY",
            "ELL",
            "LOWER",
            "UPPER",
            "WEIGHT",
        ],
        extname=ext,
        header=[
            dict(name="ELLAXIS", value=1, comment="number of angular axes"),
            dict(name="ELLAXIS1", value=axis, comment="index of angular axis 1"),
        ],
    )

    # write the metadata
    _write_metadata(fits[ext], result.dtype.metadata)


def _read_result(hdu):
    """
    Read a result array from FITS.
    """

    # read columnar data from extension
    data = hdu.read()
    h = hdu.read_header()

    # the angular axis
    elldim = h["ELLAXIS"]
    if elldim != 1:
        raise NotImplementedError("multiple angular axes are not supported")
    axis = tuple(h[f"ELLAXIS{i}"] for i in range(1, elldim + 1))

    # get data array and move axis back to right position
    result = np.moveaxis(data["ARRAY"], tuple(range(elldim)), axis)

    # construct result array with ancillary arrays and metadata
    return Result(
        result,
        axis=axis[0] if elldim == 1 else axis,
        ell=data["ELL"],
        lower=data["LOWER"],
        upper=data["UPPER"],
        weight=data["WEIGHT"],
    ).view(np.dtype(result.dtype, metadata=_read_metadata(hdu)))


def read_vmap(filename, nside=None, field=0, *, transform=False, lmax=None):
    """read visibility map from a HEALPix map file"""

    import healpy as hp

    vmap = hp.read_map(filename, field=field, dtype=float)

    # set unseen pixels to zero
    vmap[vmap == hp.UNSEEN] = 0.0

    if nside is not None and nside != hp.get_nside(vmap):
        # vmap is provided at a different resolution
        warn(f"{filename}: changing NSIDE to {nside}")
        vmap = hp.ud_grade(vmap, nside)

    if transform:
        nside = hp.get_nside(vmap)
        vmap = hp.map2alm(vmap, lmax=lmax, use_pixel_weights=True)
        pw = hp.pixwin(nside, lmax=lmax)
        hp.almxfl(vmap, 1 / pw, inplace=True)

    return vmap


def write_maps(
    filename,
    maps,
    *,
    clobber=False,
    workdir=".",
    include=None,
    exclude=None,
):
    """write a set of maps to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d maps to %s", len(maps), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, m in maps.items():
            # skip if not selected
            if not toc_match(key, include=include, exclude=exclude):
                continue

            logger.info("writing map %s", key)

            # extension name
            ext = _get_next_extname(fits, "MAP")

            # write the map in HEALPix FITS format
            _write_map(fits, ext, key, m)

    logger.info("done with %d maps", len(maps))


def read_maps(filename, workdir=".", *, include=None, exclude=None):
    """read a set of maps from a FITS file"""

    logger.info("reading maps from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of maps
    maps = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, "MAP", include=include, exclude=exclude):
        logger.info("reading map %s", key)
        maps[key] = _read_map(hdu)

    logger.info("done with %d maps", len(maps))

    # return the dictionary of maps
    return maps


def write_alms(
    filename,
    alms,
    *,
    clobber=False,
    workdir=".",
    include=None,
    exclude=None,
):
    """write a set of alms to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d alms to %s", len(alms), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, alm in alms.items():
            # skip if not selected
            if not toc_match(key, include=include, exclude=exclude):
                continue

            logger.info("writing alm %s", key)

            # extension name
            ext = _get_next_extname(fits, "ALM")

            # write the alm as structured data with metadata
            _write_complex(fits, ext, key, alm)

    logger.info("done with %d alms", len(alms))


def read_alms(filename, workdir=".", *, include=None, exclude=None):
    """read a set of alms from a FITS file"""

    logger.info("reading alms from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of alms
    alms = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, "ALM", include=include, exclude=exclude):
        logger.info("reading alm %s", key)
        alms[key] = _read_complex(hdu)

    logger.info("done with %d alms", len(alms))

    # return the dictionary of alms
    return alms


def write(path, results, *, clobber=False):
    """
    Write a set of results to FITS file.

    If the output file exists, the new results will be appended, unless
    the ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d results to %s", len(results), path)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, result in results.items():
            logger.info("writing result %s", key)

            # extension name
            ext = _string_from_key(key)

            # write the data in structured format
            _write_result(fits, ext, key, result)

    logger.info("done with %d results", len(results))


def read(path):
    """
    Read a set of results from a FITS file.
    """

    logger.info("reading results from %s", path)

    # the returned set of cls
    results = {}

    # read all HDUs in file into dict keys
    with fitsio.FITS(path) as fits:
        for hdu in fits:
            if not hdu.has_data():
                continue
            ext = hdu.get_extname()
            if not ext:
                continue
            key = _key_from_string(ext)
            if not key:
                continue
            logger.info("reading result %s", key)
            results[key] = _read_result(hdu)

    logger.info("done with %d results", len(results))

    return results


def write_cov(filename, cov, clobber=False, workdir=".", include=None, exclude=None):
    """write a set of covariance matrices to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d covariances to %s", len(cov), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, mat in cov.items():
            # skip if not selected
            if not toc_match(key, include=include, exclude=exclude):
                continue

            logger.info("writing covariance matrix %s", key)

            # extension name
            ext = _get_next_extname(fits, "COV")

            # write the covariance matrix as an image
            fits.write_image(mat, extname=ext)

            # write the key
            _write_key(fits[ext], key)

            # write the WCS
            fits[ext].write_key("WCSAXES", 2)
            fits[ext].write_key("CNAME1", "L_1")
            fits[ext].write_key("CNAME2", "L_2")
            fits[ext].write_key("CTYPE1", " ")
            fits[ext].write_key("CTYPE2", " ")
            fits[ext].write_key("CUNIT1", " ")
            fits[ext].write_key("CUNIT2", " ")

            # write the metadata
            _write_metadata(fits[ext], mat.dtype.metadata)

    logger.info("done with %d covariance(s)", len(cov))


def read_cov(filename, workdir=".", *, include=None, exclude=None):
    """read a set of covariances matrices from a FITS file"""

    logger.info("reading covariance matrices from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of covariances
    cov = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, "COV", include=include, exclude=exclude):
        logger.info("reading covariance matrix %s", key)

        # read the covariance matrix from the extension
        mat = hdu.read()

        # read and attach metadata
        mat.dtype = np.dtype(mat.dtype, metadata=_read_metadata(hdu))

        # store in set
        cov[key] = mat

    logger.info("done with %d covariance(s)", len(cov))

    # return the toc dict of covariances
    return cov


class TocFits(MutableMapping):
    """A FITS-backed TocDict."""

    tag = "EXT"
    """Tag for FITS extensions."""

    @staticmethod
    def reader(hdu):
        """Read data from FITS extension."""
        return hdu.read()

    @staticmethod
    def writer(fits, ext, key, data):
        """Write data to FITS extension."""
        if data.dtype.names is None:
            msg = "data must be structured array"
            raise TypeError(msg)
        fits.write_table(data, extname=ext)
        _write_key(fits[ext], key)

    @property
    def fits(self):
        """Return an opened FITS context manager."""
        return fitsio.FITS(self.path, mode="rw", clobber=False)

    @property
    def toc(self):
        """Return a view of the FITS table of contents."""
        return MappingProxyType(self._toc)

    def __init__(self, path, *, clobber=False):
        self.path = Path(path)

        # if new or overwriting, create an empty FITS with primary HDU
        if not self.path.exists() or clobber:
            with fitsio.FITS(self.path, mode="rw", clobber=True) as fits:
                fits.write(None)

        # create a dictionary of existing data
        self._toc = TocDict(
            {key: hdu.get_extname() for key, hdu in _iterfits(self.path, self.tag)},
        )

        # set up a weakly-referenced cache for extension data
        self._cache = WeakValueDictionary()

    def __len__(self):
        return len(self._toc)

    def __iter__(self):
        return iter(self._toc)

    def __contains__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return key in self._toc

    def __getitem__(self, key):
        ext = self._toc[key]

        # if a TocDict is returned, we have the result of a selection
        if isinstance(ext, TocDict):
            # make a new instance and copy attributes
            selected = object.__new__(self.__class__)
            selected.path = self.path
            # shared cache since both instances read the same file
            selected._cache = self._cache
            # the new toc contains the result of the selection
            selected._toc = ext
            return selected

        # a specific extension was requested, fetch data
        data = self._cache.get(ext)
        if data is None:
            with self.fits as fits:
                data = self.reader(fits[ext])
            self._cache[ext] = data
        return data

    def __setitem__(self, key, value):
        # keys are always tuples
        if not isinstance(key, tuple):
            key = (key,)

        with self.fits as fits:
            # check if an extension with the given key already exists
            # otherwise, get the first free extension with the given tag
            if key in self._toc:
                ext = self._toc[key]
            else:
                ext = _get_next_extname(fits, self.tag)

            # write data using the class writer, and update ToC as necessary
            self.writer(fits, ext, key, value)
            if key not in self._toc:
                self._toc[key] = ext

    def __delitem__(self, key):
        # fitsio does not support deletion of extensions
        msg = "deleting FITS extensions is not supported"
        raise NotImplementedError(msg)


class MapFits(TocFits):
    """FITS-backed mapping for maps."""

    tag = "MAP"
    reader = staticmethod(_read_map)
    writer = staticmethod(_write_map)


class AlmFits(TocFits):
    """FITS-backed mapping for alms."""

    tag = "ALM"
    reader = staticmethod(_read_complex)
    writer = staticmethod(_write_complex)
