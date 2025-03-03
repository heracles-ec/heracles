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
from typing import TYPE_CHECKING, Union
from warnings import warn
from weakref import WeakValueDictionary

import fitsio
import numpy as np

from .core import toc_match
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


def _write_map(fits, ext, m, *, names=None):
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
    from numpy.lib.recfunctions import structured_to_unstructured

    # read the map from the extension
    m = hdu.read()

    # turn the structured array of columns into an unstructured array
    # transpose so that columns become rows (as that is how maps are)
    # then squeeze out degenerate axes
    m = np.squeeze(structured_to_unstructured(m).T)

    # read and attach metadata
    m.dtype = np.dtype(m.dtype, metadata=_read_metadata(hdu))

    return m


def _write_complex(fits, ext, arr):
    """write complex-valued data to FITS table"""
    # deal with extra dimensions by moving last axis to first
    if arr.ndim > 1:
        arr = np.moveaxis(arr, -1, 0)

    # write the data
    fits.write_table([arr.real, arr.imag], names=["real", "imag"], extname=ext)

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


def _prepare_result_array(arr, order, size):
    """Prepare result array for writing."""

    if len(order) == 1:
        return arr[0]
    return np.transpose([np.pad(arr[i], (0, size - arr[i].size)) for i in order])


def _write_result(fits, ext, result):
    """
    Write a result array to FITS.
    """

    from heracles.result import normalize_result_axis, get_result_array

    # original unsorted ell and axis
    _ell = getattr(result, "ell", None)
    _axis = normalize_result_axis(getattr(result, "axis", None), result, _ell)

    # get decreasing order of ell axes in terms of dimension size
    order = np.argsort([result.shape[i] for i in _axis])[::-1]

    # get axis in new order
    axis = tuple(_axis[i] for i in order)

    # get array & move ell axes to front, largest first
    arr = np.moveaxis(result, axis, tuple(range(len(axis))))

    # length of largest axis will be the number of rows
    nrows = arr.shape[0]

    # get data arrays
    ell = _prepare_result_array(get_result_array(result, "ell"), order, nrows)
    lower = _prepare_result_array(get_result_array(result, "lower"), order, nrows)
    upper = _prepare_result_array(get_result_array(result, "upper"), order, nrows)
    weight = _prepare_result_array(get_result_array(result, "weight"), order, nrows)

    # construct the result header
    kw_ellaxis = str(axis).replace(" ", "")
    header = [
        dict(name="ELLAXIS", value=kw_ellaxis, comment="angular axis indices"),
    ]

    # write the result as columnar data
    fits.write_table(
        [
            arr,
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
        header=header,
    )

    # write the metadata
    _write_metadata(fits[ext], result.dtype.metadata)


def _read_result(hdu):
    """
    Read a result array from FITS.
    """

    from ast import literal_eval

    # read columnar data from extension
    data = hdu.read()
    h = hdu.read_header()

    # the angular axis
    axis = literal_eval(h["ELLAXIS"])

    # get data array and move axis back to right position
    arr = np.moveaxis(data["ARRAY"], tuple(range(len(axis))), axis)

    # sort ell axes into natural order
    order = np.argsort(axis)

    # get ells
    _ell = data["ELL"]
    if _ell.ndim == 1:
        ell = _ell
    else:
        ell = tuple(_ell[: arr.shape[axis[i]], i] for i in order)

    # get lower bounds
    _lower = data["LOWER"]
    if _lower.ndim == 1:
        lower = _lower
    else:
        lower = tuple(_lower[: arr.shape[axis[i]], i] for i in order)

    # get upper bounds
    _upper = data["UPPER"]
    if _upper.ndim == 1:
        upper = _upper
    else:
        upper = tuple(_upper[: arr.shape[axis[i]], i] for i in order)

    # get weights
    _weight = data["WEIGHT"]
    if _weight.ndim == 1:
        weight = _weight
    else:
        weight = tuple(_weight[: arr.shape[axis[i]], i] for i in order)

    # construct result array with ancillary arrays and metadata
    return Result(
        arr.view(np.dtype(arr.dtype, metadata=_read_metadata(hdu))),
        axis=tuple(axis[i] for i in order),
        ell=ell,
        lower=lower,
        upper=upper,
        weight=weight,
    )


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


def write_maps(path, maps, *, clobber=False):
    """write a set of maps to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d maps to %s", len(maps), path)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, m in maps.items():
            logger.info("writing map %s", key)

            # extension name
            ext = _string_from_key(key)

            # write the map in HEALPix FITS format
            _write_map(fits, ext, m)

    logger.info("done with %d maps", len(maps))


def read_maps(path, *, include=None, exclude=None):
    """read a set of maps from a FITS file"""

    logger.info("reading maps from %s", path)

    # the returned set of maps
    maps = {}

    # read all HDUs in file into dict keys
    with fitsio.FITS(path) as fits:
        for hdu in fits:
            # skip extensions with no data
            if not hdu.has_data():
                continue
            # get extension name, skip if empty (= no key)
            ext = hdu.get_extname()
            if not ext:
                continue
            # decode extension name into key, skip if empty
            key = _key_from_string(ext)
            if not key:
                continue
            # match key against explicit include and exclude
            if not toc_match(key, include=include, exclude=exclude):
                continue
            logger.info("reading map %s", key)
            maps[key] = _read_map(hdu)

    logger.info("done with %d maps", len(maps))

    # return the dictionary of maps
    return maps


def write_alms(path, alms, *, clobber=False):
    """write a set of alms to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d alms to %s", len(alms), path)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, alm in alms.items():
            logger.info("writing alm %s", key)

            # extension name
            ext = _string_from_key(key)

            # write the alm as structured data with metadata
            _write_complex(fits, ext, alm)

    logger.info("done with %d alms", len(alms))


def read_alms(path, *, include=None, exclude=None):
    """read a set of alms from a FITS file"""

    logger.info("reading alms from %s", path)

    # the returned set of alms
    alms = {}

    # read all HDUs in file into dict keys
    with fitsio.FITS(path) as fits:
        for hdu in fits:
            # skip extensions with no data
            if not hdu.has_data():
                continue
            # get extension name, skip if empty (= no key)
            ext = hdu.get_extname()
            if not ext:
                continue
            # decode extension name into key, skip if empty
            key = _key_from_string(ext)
            if not key:
                continue
            # match key against explicit include and exclude
            if not toc_match(key, include=include, exclude=exclude):
                continue
            logger.info("reading alm %s", key)
            alms[key] = _read_complex(hdu)

    logger.info("done with %d alms", len(alms))

    # return the dictionary of alms
    return alms


def dict2mat(cls, cov):
    Clkeys = list(cls.keys())
    ncls = len(Clkeys)
    nells = [len(cls[key].ell) for key in Clkeys]
    full_cov = np.zeros((np.sum(nells), np.sum(nells)))
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            full_cov[i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j] = (
                cov[covkey]
            )
            if i != j:
                full_cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ] = cov[covkey].T
    return full_cov


def mat2dict(cls, cov):
    Clkeys = list(cls.keys())
    ncls = len(Clkeys)
    nells = [len(cls[key].ell) for key in Clkeys]
    Cl_cov_dict = {}
    for i in range(0, len(Clkeys)):
        for j in range(i, len(Clkeys)):
            ki = Clkeys[i]
            kj = Clkeys[j]
            A, B, nA, nB = ki[0], ki[1], ki[2], ki[3]
            C, D, nC, nD = kj[0], kj[1], kj[2], kj[3]
            covkey = (A, B, C, D, nA, nB, nC, nD)
            size_i = nells[i]
            size_j = nells[j]
            Cl_cov_dict[covkey] = cov[
                i * size_i : (i + 1) * size_i, j * size_j : (j + 1) * size_j
            ]
            if i != j:
                Cl_cov_dict[covkey] = cov[
                    j * size_j : (j + 1) * size_j, i * size_i : (i + 1) * size_i
                ]
    return Cl_cov_dict


def split_comps(covkey, ncls1, ncls2):
    a1, b1, a2, b2, i1, j1, i2, j2 = covkey
    if ncls1 == 1:
        f1 = [("POS", "POS")]
    elif ncls1 == 2:
        f1 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls1 == 3:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f1 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    if ncls2 == 1:
        f2 = [("POS", "POS")]
    elif ncls2 == 2:
        f2 = [("POS", "G_E"), ("POS", "G_B")]
    elif ncls2 == 3:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B")]
    elif ncls2 == 4:
        f2 = [("G_E", "G_E"), ("G_B", "G_B"), ("G_E", "G_B"), ("G_B", "G_E")]

    covkeys = {}
    for i in range(ncls1):
        for j in range(ncls2):
            _f1 = f1[i]
            _f2 = f2[j]
            _a1, _b1 = _f1
            _a2, _b2 = _f2
            if _a2 == "G_B" and _b2 == "G_E":
                _a2, _b2 = "G_E", "G_B"
                i2, j2 = j2, i2
            _covkey = _a1, _b1, _a2, _b2, i1, j1, i2, j2
            covkeys[(i, j)] = _covkey
    return covkeys, f1, f2


def cov2spinblocks(cls, cov):
    _covs = {}
    cls_keys = list(cls.keys())
    for i in range(0, len(cls_keys)):
        for j in range(i, len(cls_keys)):
            k1 = cls_keys[i]
            k2 = cls_keys[j]
            ell1 = cls[k1].ell
            ell2 = cls[k2].ell
            cl1 = np.atleast_2d(cls[k1])
            cl2 = np.atleast_2d(cls[k2])
            ncls1, nells1 = cl1.shape
            ncls2, nells2 = cl2.shape
            A, B, nA, nB = k1[0], k1[1], k1[2], k1[3]
            C, D, nC, nD = k2[0], k2[1], k2[2], k2[3]

            covkey = (A, B, C, D, nA, nB, nC, nD)
            # Writes the covkeys of the spin components associated with covkey
            # it also returns what fields go in what axis
            #  comps1     (E, E) (B, B) (E,B) <--- comps2
            # (POS, E)
            # (POS, B)
            covkeys, comps1, comps2 = split_comps(covkey, ncls1, ncls2)
            # Save comps in dtype metadata
            dt = np.dtype(
                float,
                metadata={
                    "fields1": comps1,
                    "fields2": comps2,
                },
            )
            _cov = np.zeros((ncls1, ncls2, nells1, nells2), dtype=dt)
            for i in range(ncls1):
                for j in range(ncls2):
                    _covkey = covkeys[(i, j)]
                    if _covkey not in cov.keys():
                        # This triggers if the element doesn't exist
                        # but the symmetrical term does
                        _cov[i, j, :, :] = np.zeros((nells2, nells1))
                    else:
                        _cov[i, j, :, :] = cov[_covkey]
            _covs[covkey] = Result(_cov, ell=(ell1, ell2))
    return _covs


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
            _write_result(fits, ext, result)

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


class FitsDict(MutableMapping):
    """A FITS-backed mapping."""

    @staticmethod
    def reader(hdu):
        """Read data from FITS extension."""
        return hdu.read()

    @staticmethod
    def writer(fits, ext, data):
        """Write data to FITS extension."""
        if data.dtype.names is None:
            msg = "data must be structured array"
            raise TypeError(msg)
        fits.write_table(data, extname=ext)

    @property
    def fits(self):
        """Return an opened FITS context manager."""
        return fitsio.FITS(self.path, mode="rw", clobber=False)

    def __init__(self, path, *, clobber=False):
        self.path = Path(path)

        # if new or overwriting, create an empty FITS with primary HDU
        if not self.path.exists() or clobber:
            with fitsio.FITS(self.path, mode="rw", clobber=True) as fits:
                fits.write(None)

        # set up a weakly-referenced cache for extension data
        self._cache = WeakValueDictionary()

    def __iter__(self):
        with fitsio.FITS(self.path) as fits:
            for hdu in fits:
                # skip extensions with no data
                if not hdu.has_data():
                    continue
                # get extension name, skip if empty (= no key)
                ext = hdu.get_extname()
                if not ext:
                    continue
                # decode extension name into key, skip if empty
                key = _key_from_string(ext)
                if not key:
                    continue
                yield key

    def __len__(self):
        n = 0
        for _ in iter(self):
            n += 1
        return n

    def __contains__(self, key):
        ext = _string_from_key(key)
        with fitsio.FITS(self.path) as fits:
            return ext in fits

    def __getitem__(self, key):
        # a specific extension was requested, fetch data
        ext = _string_from_key(key)
        data = self._cache.get(ext)
        if data is None:
            with self.fits as fits:
                data = self.reader(fits[ext])
            self._cache[ext] = data
        return data

    def __setitem__(self, key, value):
        ext = _string_from_key(key)
        with self.fits as fits:
            self.writer(fits, ext, value)

    def __delitem__(self, key):
        # fitsio does not support deletion of extensions
        msg = "deleting FITS extensions is not supported"
        raise NotImplementedError(msg)


class MapFits(FitsDict):
    """FITS-backed mapping for maps."""

    reader = staticmethod(_read_map)
    writer = staticmethod(_write_map)


class AlmFits(FitsDict):
    """FITS-backed mapping for alms."""

    reader = staticmethod(_read_complex)
    writer = staticmethod(_write_complex)
