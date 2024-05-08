"""
Core (:mod:`heracles.core`)
============================

.. currentmodule:: heracles.core

The :mod:`heracles.core` module provides common core functionality

.. autofunction:: toc_match
.. autofunction:: toc_filter
.. autofunction:: TocDict
.. autofunction:: update_metadata

"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping, Sequence
from typing import TypeVar

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
