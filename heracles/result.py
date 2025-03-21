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

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray, DTypeLike


def normalize_result_axis(axis, result, ell):
    """Return an axis tuple for a result."""
    try:
        from numpy.lib.array_utils import normalize_axis_tuple
    except ModuleNotFoundError:
        from numpy.lib.stride_tricks import normalize_axis_tuple

    if axis is None:
        if result.ndim == 0:
            axis = ()
        elif isinstance(ell, tuple):
            axis = tuple(range(-len(ell), 0))
        else:
            axis = -1
    return normalize_axis_tuple(axis, result.ndim, "axis")


def get_result_array(result, name):
    """Return a normalised version of the array *name* from *result*."""

    arr = getattr(result, name, None)
    axis = normalize_result_axis(getattr(result, "axis", None), result, arr)
    if arr is None:
        if name == "ell":
            arr = tuple(np.arange(result.shape[i]) for i in axis)
        elif name == "lower":
            arr = get_result_array(result, "ell")
        elif name == "upper":
            _lower = get_result_array(result, "lower")
            arr = tuple(np.append(lo[1:], lo[-1] + 1) for lo in _lower)
        elif name == "weight":
            arr = tuple(np.ones(result.shape[i]) for i in axis)
        else:
            raise ValueError(f"cannot make default for array {name!r}")
    if isinstance(arr, tuple):
        return arr
    return (arr,) * len(axis)


@dataclass(frozen=True, repr=False)
class Result:
    """
    Container for results.
    """

    array: NDArray[Any]
    ell: NDArray[Any] | tuple[NDArray[Any], ...] | None = None
    axis: int | tuple[int, ...] | None = None
    lower: NDArray[Any] | tuple[NDArray[Any], ...] | None = None
    upper: NDArray[Any] | tuple[NDArray[Any], ...] | None = None
    weight: NDArray[Any] | tuple[NDArray[Any], ...] | None = None

    def __post_init__(self) -> None:
        axis = normalize_result_axis(self.axis, self.array, self.ell)
        object.__setattr__(self, "axis", axis)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis!r})"

    def __array__(self, dtype=None, *, copy=None) -> NDArray[Any]:
        if copy is not None:
            # copy being set means NumPy v2, so it's safe to pass it on
            return self.array.__array__(dtype, copy=copy)
        # NumPy v1 might not know about copy
        return self.array.__array__(dtype)

    def __getitem__(self, key):
        return self.array[key]

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.array.dtype


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

    # shape of the data
    ellaxis = getattr(result, "axis", None)
    if ellaxis is None:
        ellaxis = (result.ndim - 1,)
    if len(ellaxis) != 1:
        # multi-binning not implemented yet
        raise NotImplementedError("only 1D binning is supported")
    axis = ellaxis[0]
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

    # construct output dtype with metadata
    md = {}
    if result.dtype.metadata:
        md.update(result.dtype.metadata)
    dt = np.dtype(float, metadata=md)

    # create an empty binned output array
    sh = shape[:axis] + (m - 1,) + shape[axis + 1 :]
    out = np.empty(sh, dtype=dt)

    # compute the binned result axis by axis
    for i in np.ndindex(shape[:axis]):
        for j in np.ndindex(shape[axis + 1 :]):
            k = (*i, slice(None), *j)
            out[k] = norm(np.bincount(index, w * result[k], m)[1:m], wb)

    # compute the binned ell
    ell = norm(np.bincount(index, w * ell, m)[1:m], wb)

    # construct the result
    return Result(
        out,
        ell=ell,
        axis=axis,
        lower=bins[:-1],
        upper=bins[1:],
        weight=wb,
    )
