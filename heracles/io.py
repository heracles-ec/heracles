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
"""module for file reading and writing"""

import logging
import os
from collections.abc import MutableMapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from warnings import warn
from weakref import WeakValueDictionary

import fitsio
import numpy as np

from .core import TocDict, toc_match

logger = logging.getLogger(__name__)


_METADATA_COMMENTS = {
    "catalog": "catalog of field",
    "catalog_1": "catalog of first field",
    "catalog_2": "catalog of second field",
    "spin": "spin weight of field",
    "spin1_": "spin weight of first field",
    "spin2_": "spin weight of second field",
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


def _extname_from_key(key):
    """
    Return FITS extension name for a given key.
    """
    if not isinstance(key, tuple):
        key = (key,)
    return ",".join(map(str, key))


def _key_from_extname(ext):
    """
    Return key for a given FITS extension name.
    """
    return tuple(int(s) if s.isdigit() else s for s in ext.split(","))


def _iterfits(path, include=None, exclude=None):
    """
    Iterate over HDUs that correspond to valid keys.
    """
    with fitsio.FITS(path) as fits:
        for hdu in fits:
            extname = hdu.get_extname()
            if not extname:
                continue
            key = _key_from_extname(extname)
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

    # read the map from the extension
    m = hdu.read()

    # turn the structured array of columns into an unstructured array
    # transpose so that columns become rows (as that is how maps are)
    # then squeeze out degenerate axes
    m = np.squeeze(np.lib.recfunctions.structured_to_unstructured(m).T)

    # read and attach metadata
    m.dtype = np.dtype(m.dtype, metadata=_read_metadata(hdu))

    return m


def _write_complex(fits, ext, arr):
    """write complex-valued data to FITS table"""
    # write the data
    fits.write_table([arr.real, arr.imag], names=["real", "imag"], extname=ext)

    # write the metadata
    _write_metadata(fits[ext], arr.dtype.metadata)


def _read_complex(hdu):
    """read complex-valued data from FITS table"""
    # read structured data as complex array
    raw = hdu.read()
    arr = np.empty(len(raw), dtype=complex)
    arr.real = raw["real"]
    arr.imag = raw["imag"]
    del raw
    # read and attach metadata
    arr.dtype = np.dtype(arr.dtype, metadata=_read_metadata(hdu))
    return arr


def _write_twopoint(fits, ext, arr, name):
    """convert two-point data (i.e. one L column) to structured array"""

    arr = np.asanyarray(arr)

    # get the data into structured array if not already
    if arr.dtype.names is None:
        n, *dims = arr.shape
        data = arr

        dt = np.dtype(
            [
                ("L", float),
                (name, arr.dtype.str, dims) if dims else (name, arr.dtype.str),
                ("LMIN", float),
                ("LMAX", float),
                ("W", float),
            ],
            metadata=dict(arr.dtype.metadata or {}),
        )

        arr = np.empty(n, dt)
        arr["L"] = np.arange(n)
        arr[name] = data
        arr["LMIN"] = arr["L"]
        arr["LMAX"] = arr["L"] + 1
        arr["W"] = 1

    # write the twopoint data
    fits.write_table(arr, extname=ext)

    # write the metadata
    _write_metadata(fits[ext], arr.dtype.metadata)


def _read_twopoint(hdu):
    """read two-point data from FITS"""
    # read data from extension
    arr = hdu.read()
    # read and attach metadata
    arr.dtype = np.dtype(arr.dtype, metadata=_read_metadata(hdu))
    return arr


def read_vmap(filename, nside=None, field=0):
    """read visibility map from a HEALPix map file"""

    import healpy as hp

    vmap = hp.read_map(filename, field=field, dtype=float)

    # set unseen pixels to zero
    vmap[vmap == hp.UNSEEN] = 0

    if nside is not None and nside != hp.get_nside(vmap):
        # vmap is provided at a different resolution
        warn(f"{filename}: changing NSIDE to {nside}")
        vmap = hp.ud_grade(vmap, nside)

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

            # the cl extension name
            ext = _extname_from_key(key)

            # write the map in HEALPix FITS format
            _write_map(fits, ext, m)

    logger.info("done with %d maps", len(maps))


def read_maps(filename, workdir=".", *, include=None, exclude=None):
    """read a set of maps from a FITS file"""

    logger.info("reading maps from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of maps
    maps = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, include=include, exclude=exclude):
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

            # the cl extension name
            ext = _extname_from_key(key)

            # write the alm as structured data with metadata
            _write_complex(fits, ext, alm)

    logger.info("done with %d alms", len(alms))


def read_alms(filename, workdir=".", *, include=None, exclude=None):
    """read a set of alms from a FITS file"""

    logger.info("reading alms from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of alms
    alms = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, include=include, exclude=exclude):
        logger.info("reading alm %s", key)
        alms[key] = _read_complex(hdu)

    logger.info("done with %d alms", len(alms))

    # return the dictionary of alms
    return alms


def write_cls(filename, cls, *, clobber=False, workdir=".", include=None, exclude=None):
    """write a set of cls to FITS file

    If the output file exists, the new estimates will be appended, unless the
    ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d cls to %s", len(cls), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, cl in cls.items():
            # skip if not selected
            if not toc_match(key, include=include, exclude=exclude):
                continue

            logger.info("writing cl %s", key)

            # the cl extension name
            ext = _extname_from_key(key)

            # write the data in structured format
            _write_twopoint(fits, ext, cl, "CL")

    logger.info("done with %d cls", len(cls))


def read_cls(filename, workdir=".", *, include=None, exclude=None):
    """read a set of cls from a FITS file"""

    logger.info("reading cls from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of cls
    cls = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, include=include, exclude=exclude):
        logger.info("reading cl %s", key)
        cls[key] = _read_twopoint(hdu)

    logger.info("done with %d cls", len(cls))

    # return the dictionary of cls
    return cls


def write_mms(filename, mms, *, clobber=False, workdir=".", include=None, exclude=None):
    """write a set of mixing matrices to FITS file

    If the output file exists, the new mixing matrices will be appended, unless
    the ``clobber`` parameter is set to ``True``.

    """

    logger.info("writing %d mm(s) to %s", len(mms), filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # if new or overwriting, create an empty FITS with primary HDU
    if not os.path.isfile(path) or clobber:
        with fitsio.FITS(path, mode="rw", clobber=True) as fits:
            fits.write(None)

    # reopen FITS for writing data
    with fitsio.FITS(path, mode="rw", clobber=False) as fits:
        for key, mm in mms.items():
            # skip if not selected
            if not toc_match(key, include=include, exclude=exclude):
                continue

            logger.info("writing mm %s", key)

            # the mm extension name
            ext = _extname_from_key(key)

            # write the data in structured format
            _write_twopoint(fits, ext, mm, "MM")

    logger.info("done with %d mm(s)", len(mms))


def read_mms(filename, workdir=".", *, include=None, exclude=None):
    """read a set of mixing matrices from a FITS file"""

    logger.info("reading mixing matrices from %s", filename)

    # full path to FITS file
    path = os.path.join(workdir, filename)

    # the returned set of mms
    mms = TocDict()

    # iterate over valid HDUs in the file
    for key, hdu in _iterfits(path, include=include, exclude=exclude):
        logger.info("writing mm %s", key)
        mms[key] = _read_twopoint(hdu)

    logger.info("done with %d mm(s)", len(mms))

    # return the dictionary of mms
    return mms


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
        for (k1, k2), mat in cov.items():
            # skip if not selected
            if not toc_match((k1, k2), include=include, exclude=exclude):
                continue

            # the cl extension name
            ext = _extname_from_key(k1 + k2)

            logger.info("writing %s x %s covariance matrix", k1, k2)

            # write the covariance matrix as an image
            fits.write_image(mat, extname=ext)

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
    for key, hdu in _iterfits(path, include=include, exclude=exclude):
        k1, k2 = key[: len(key) // 2], key[len(key) // 2 :]

        logger.info("reading %s x %s covariance matrix", k1, k2)

        # read the covariance matrix from the extension
        mat = hdu.read()

        # read and attach metadata
        mat.dtype = np.dtype(mat.dtype, metadata=_read_metadata(hdu))

        # store in set
        cov[k1, k2] = mat

    logger.info("done with %d covariance(s)", len(cov))

    # return the toc dict of covariances
    return cov


class TocFits(MutableMapping):
    """A FITS-backed TocDict."""

    @staticmethod
    def reader(fits, ext):
        """Read data from FITS extension."""
        return fits[ext].read()

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

        # read extensions
        with self.fits as fits:
            self._toc = TocDict()
            for hdu in fits:
                if ext := hdu.get_extname():
                    self._toc[_key_from_extname(ext)] = ext

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
                data = self.reader(fits, ext)
            self._cache[ext] = data
        return data

    def __setitem__(self, key, value):
        # keys are always tuples
        if not isinstance(key, tuple):
            key = (key,)

        # check if an extension with the given key already exists
        # otherwise, get the first free extension with the given tag
        if key in self._toc:
            ext = self._toc[key]
        else:
            ext = _extname_from_key(key)

        # write data using the class writer, and update ToC as necessary
        with self.fits as fits:
            self.writer(fits, ext, value)
            if key not in self._toc:
                self._toc[key] = ext

    def __delitem__(self, key):
        # fitsio does not support deletion of extensions
        msg = "deleting FITS extensions is not supported"
        raise NotImplementedError(msg)


class MapFits(TocFits):
    """FITS-backed mapping for maps."""

    reader = staticmethod(_read_map)
    writer = staticmethod(_write_map)


class AlmFits(TocFits):
    """FITS-backed mapping for alms."""

    reader = staticmethod(_read_complex)
    writer = staticmethod(_write_complex)


class ClsFits(TocFits):
    """FITS-backed mapping for cls."""

    reader = staticmethod(_read_twopoint)
    writer = partial(_write_twopoint, name="CL")


class MmsFits(TocFits):
    """FITS-backed mapping for mixing matrices."""

    reader = staticmethod(_read_twopoint)
    writer = partial(_write_twopoint, name="MM")
