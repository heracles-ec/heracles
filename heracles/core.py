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
"""
Module for common core functionality.
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping, Sequence
from typing import TypeVar

import healpy as hp
import numpy as np

T = TypeVar("T")


def toc_match(key, include=None, exclude=None):
    """return whether a tocdict entry matches include/exclude criteria"""
    if not isinstance(key, tuple):
        key = (key,)
    if include is not None:
        for pattern in include:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                break
        else:
            return False
    if exclude is not None:
        for pattern in exclude:
            if all(p is Ellipsis or p == k for p, k in zip(pattern, key)):
                return False
    return True


def toc_filter(obj, include=None, exclude=None):
    """return a filtered toc dict ``d``"""
    if isinstance(obj, Sequence):
        return [toc_filter(item, include, exclude) for item in obj]
    if isinstance(obj, Mapping):
        return {k: v for k, v in obj.items() if toc_match(k, include, exclude)}
    msg = "invalid input type"
    raise TypeError(msg)


# subclassing UserDict here since that returns the correct type from methods
# such as __copy__(), __or__(), etc.
class TocDict(UserDict):
    """Table-of-contents dictionary with pattern-based lookup"""

    def __getitem__(self, pattern):
        """look up one or many keys in dict"""
        # first, see if pattern is a valid entry in the dict
        # might fail with KeyError (no such entry) or TypeError (not hashable)
        try:
            return self.data[pattern]
        except (KeyError, TypeError):
            pass
        # pattern might be a single object such as e.g. "X"
        if not isinstance(pattern, tuple):
            pattern = (pattern,)
        # no pattern == matches everything
        if not pattern:
            return self.copy()
        # go through all keys in the dict and match them against the pattern
        # return an object of the same type
        found = self.__class__()
        for key, value in self.data.items():
            if isinstance(key, tuple):
                # key too short, cannot possibly match pattern
                if len(key) < len(pattern):
                    continue
                # match every part of pattern against the given key
                # Ellipsis (...) is a wildcard and comparison is skipped
                if all(p == k for p, k in zip(pattern, key) if p is not ...):
                    found[key] = value
            else:
                # key is a single entry, pattern must match it
                if pattern == (...,) or pattern == (key,):
                    found[key] = value
        # nothing matched the pattern, treat as usual dict lookup error
        if not found:
            raise KeyError(pattern)
        return found


def update_metadata(array, *sources, **metadata):
    """update metadata of an array dtype"""
    md = {}
    if array.dtype.metadata is not None:
        md.update(array.dtype.metadata)
    for source in sources:
        md.update(source.metadata)
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


def add_metadata_to_external_map(
    m,
    spin,
    verbose=True,
    geometry="healpix",
    kernel="healpix",
    deconv=True,
    catalog=None,
    ngal=1.0,
    nbar=1.0,
    wmean=1.0,
    bias=0.0,
    var=1.0,
):
    # Basic checks
    if spin == 0 and len(m.shape) != 1:
        raise ValueError("Spin-0 map must be 1D array")
    if spin == 2 and len(m.shape) != 2:
        raise ValueError("Spin-2 map must be 2D array with shape (2, npix)")
    # Derived quantities
    nside = hp.get_nside(m)
    if spin == 0:
        fsky = len(m[m != 0]) / len(m)
    else:
        fsky = len(m[0][m[0] != 0]) / len(m[0])
    variance = var / wmean**2
    neff = ngal / (4 * np.pi * fsky)
    # update metadata
    if verbose:
        print("Adding metadata to external map:")
        print(f"  geometry: {geometry}")
        print(f"  kernel: {kernel}")
        print(f"  deconv: {deconv}")
        print(f"  catalog: {catalog}")
        print(f"  nside: {nside}")
        print(f"  ngal: {ngal}")
        print(f"  nbar: {nbar}")
        print(f"  wmean: {wmean}")
        print(f"  bias: {bias}")
        print(f"  var: {var}")
        print(f"  variance: {variance}")
        print(f"  neff: {neff}")
        print(f"  fsky: {fsky}")
        print(f"  spin: {spin}")
    update_metadata(
        m,
        geometry=geometry,
        kernel=kernel,
        deconv=deconv,
        catalog=catalog,
        nside=nside,
        ngal=ngal,
        nbar=nbar,
        wmean=wmean,
        bias=bias,
        var=var,
        variance=variance,
        neff=neff,
        fsky=fsky,
        spin=spin,
    )
    return m


class ExceptionExplainer:
    """
    Context manager that adds a note to exceptions.
    """

    def __init__(
        self,
        exc_type: type[BaseException] | tuple[type[BaseException], ...],
        note: str,
    ) -> None:
        self.exc_type = exc_type
        self.note = note

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type and issubclass(exc_type, self.exc_type):
            try:
                exc_value.add_note(self.note)
            except AttributeError:
                pass


external_dependency_explainer = ExceptionExplainer(
    ModuleNotFoundError,
    "You are trying to import a Heracles module that relies on a missing "
    "external dependency. These dependencies are not part of the core "
    "Heracles functionality, and are therefore not installed automatically. "
    "Please install the missing packages, and this error will disappear.",
)
