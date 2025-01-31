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
Module for the result array type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

try:
    from numpy.lib.array_utils import normalize_axis_index
except ModuleNotFoundError:
    from numpy.core.multiarray import normalize_axis_index

if TYPE_CHECKING:
    from typing import Any, Self
    from numpy.typing import NDArray


class Result(np.ndarray):
    """
    NumPy :class:`~numpy.ndarray` subclass with extra properties for
    two-point results and beyond.
    """

    __slots__ = ("axis", "ell", "lower", "upper", "weight")

    def __new__(
        cls,
        arr: NDArray[Any],
        ell: NDArray[Any] | None = None,
        *,
        axis: int | None = None,
        lower: NDArray[Any] | None = None,
        upper: NDArray[Any] | None = None,
        weight: NDArray[Any] | None = None,
    ) -> Self:
        obj = np.asarray(arr).view(cls)
        if axis is None:
            if obj.ndim == 0:
                axis = None
            else:
                axis = obj.ndim - 1
        else:
            axis = normalize_axis_index(axis, ndim=obj.ndim)
        obj.axis = axis
        obj.ell = ell
        obj.lower = lower
        obj.upper = upper
        obj.weight = weight
        return obj

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        if obj is None:
            return
        self.axis = getattr(obj, "axis", None)
        self.ell = getattr(obj, "ell", None)
        self.lower = getattr(obj, "lower", None)
        self.upper = getattr(obj, "upper", None)
        self.weight = getattr(obj, "weight", None)

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        out = super().__array_wrap__(arr, context)
        if out is self or type(self) is not Result:
            return out
        if return_scalar:
            return out.item()
        return out.view(np.ndarray)


def binned(result, bins, weight=None):
    """
    Compute binned results.
    """

    if isinstance(result, Mapping):
        return {key: binned(value, bins, weight) for key, value in result.items()}

    def norm(a, b):
        """divide a by b if a is nonzero"""
        out = np.zeros(np.broadcast(a, b).shape)
        return np.divide(a, b, where=(a != 0), out=out)

    # flatten list of bins
    bins = np.reshape(bins, -1)
    m = bins.size

    # convert result to ndarray or subclass
    # support for subclasses (Result) is important here
    result = np.asanyarray(result)

    # shape of the data
    axis = getattr(result, "axis", result.ndim - 1)
    shape = result.shape
    n = shape[axis]

    # get ell from results or assume [0, n)
    ell = getattr(result, "ell", None)
    if ell is None:
        ell = np.arange(n)

    # combine weights of result with weights from string or given array
    w = getattr(result, "weight", None)
    if w is None:
        w = np.ones(n)
    if weight is None:
        pass
    elif isinstance(weight, str):
        if weight == "l(l+1)":
            w *= ell * (ell + 1)
        elif weight == "2l+1":
            w *= 2 * ell + 1
        else:
            msg = f"unknown weights string: {weight}"
            raise ValueError(msg)
    else:
        w *= weight[:n]

    # get the bin index for each ell
    index = np.digitize(ell, bins)

    assert index.size == ell.size

    # compute the binned weight
    wb = np.bincount(index, weights=w, minlength=m)[1:m]

    # create an empty binned output array
    out = Result(np.empty(shape[:axis] + (m - 1,) + shape[axis + 1 :]), axis=axis)

    # compute the binned result axis by axis
    for i in np.ndindex(shape[:axis]):
        for j in np.ndindex(shape[axis + 1 :]):
            k = (*i, slice(None), *j)
            out[k] = norm(np.bincount(index, w * result[k], m)[1:m], wb)

    # compute the binned ell
    out.ell = norm(np.bincount(index, w * ell, m)[1:m], wb)

    # set bin edges
    out.lower = bins[:-1]
    out.upper = bins[1:]

    # set the binned weight
    out.weight = wb

    # all done
    return out
