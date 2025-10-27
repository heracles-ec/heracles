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
    spin: int | tuple[int, ...] | None = None
    axis: int | tuple[int, ...] | None = None
    lower: NDArray[Any] | tuple[NDArray[Any], ...] | None = None
    upper: NDArray[Any] | tuple[NDArray[Any], ...] | None = None
    weight: NDArray[Any] | tuple[NDArray[Any], ...] | None = None

    def __post_init__(self) -> None:
        axis = normalize_result_axis(self.axis, self.array, self.ell)
        object.__setattr__(self, "axis", axis)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis!r})"

    def __array__(
        self,
        dtype: np.dtype[Any] | None = None,
        *,
        copy: bool | None = None,
    ) -> NDArray[Any]:
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
    # get the spin if instance of Result
    spin = getattr(result, "spin", None)   

    # get ell values from result
    ells = get_result_array(result, "ell")

    # get normalised axis tuple from result
    axes = normalize_result_axis(getattr(result, "axis", None), result, ells)

    # normalise bins into a tuple if a single set is given
    if not isinstance(bins, tuple):
        bins = (bins,) * len(axes)

    # make sure length of given bins matches ell axes
    if len(bins) != len(axes):
        raise ValueError("result and bins have different number of ell axes")

    # normalise weight into a tuple if a single weight is given
    if not isinstance(weight, tuple):
        weight = (weight,) * len(axes)

    # make sure length of given weight matches ell axes
    if len(weight) != len(axes):
        raise ValueError("result and weight have different number of ell axes")

    # get existing weights from result
    result_weight = get_result_array(result, "weight")

    # combine weights of result with weights from string or given array
    combined_weight = []
    for ell, w1, w2 in zip(ells, weight, result_weight):
        if w1 is None:
            w = w2
        elif isinstance(w1, str):
            if w1 == "l(l+1)":
                w = ell * (ell + 1) * w2
            elif w1 == "2l+1":
                w = (2 * ell + 1) * w2
            else:
                msg = f"unknown weights string: {w1}"
                raise ValueError(msg)
        else:
            w = w1[: w2.size] * w2
        combined_weight.append(w)

    # construct output dtype with metadata
    md = {}
    if result.dtype.metadata:
        md.update(result.dtype.metadata)
    dt = np.dtype(float, metadata=md)

    # make a copy of the array to apply the binning
    out = np.copy(result).view(dt)

    # this will hold the binned ells and weigths
    binned_ell: tuple[NDArray[Any], ...] | NDArray[Any] = ()
    binned_weight: tuple[NDArray[Any], ...] | NDArray[Any] = ()

    # apply binning over each axis
    for axis, ell, w, b in zip(axes, ells, combined_weight, bins):
        # number of bins for this axis
        m = b.size

        # get the bin index for each ell
        index = np.digitize(ell, b)

        # compute the binned weight
        wb = np.bincount(index, weights=w, minlength=m)[1:m]

        # compute the binned ell
        ellb = norm(np.bincount(index, w * ell, m)[1:m], wb)

        # output shape, axis is turned into size m
        shape = out.shape[:axis] + (m - 1,) + out.shape[axis + 1 :]

        # create an empty binned output array
        tmp = np.empty(shape, dtype=dt)

        # compute the binned result axis by axis
        for before in np.ndindex(shape[:axis]):
            for after in np.ndindex(shape[axis + 1 :]):
                k = (*before, slice(None), *after)
                tmp[k] = norm(np.bincount(index, w * out[k], m)[1:m], wb)

        # array is now binned over axis
        out = tmp

        # store outputs
        binned_ell += (ellb,)
        binned_weight += (wb,)

    # compute bin edges
    binned_lower = tuple(b[:-1] for b in bins)
    binned_upper = tuple(b[1:] for b in bins)

    # return plain arrays when there is a single ell axis
    if len(axes) == 1:
        binned_ell = binned_ell[0]
        binned_lower = binned_lower[0]
        binned_upper = binned_upper[0]
        binned_weight = binned_weight[0]

    # construct the result
    return Result(
        out,
        spin=spin,
        ell=binned_ell,
        axis=axes,
        lower=binned_lower,
        upper=binned_upper,
        weight=binned_weight,
    )


def truncated(result, ell_max):
    """
    Truncate result arrays at given maximum ell values.
    """

    if isinstance(result, Mapping):
        return {key: truncated(value, ell_max) for key, value in result.items()}

    ells = get_result_array(result, "ell")
    axes = normalize_result_axis(getattr(result, "axis", None), result, ells)

    if not isinstance(ell_max, tuple):
        ell_max = (ell_max,) * len(axes)
    if len(ell_max) != len(axes):
        raise ValueError("result and ell_max have different number of ell axes")

    md = {}
    if result.dtype.metadata:
        md.update(result.dtype.metadata)
    dt = np.dtype(float, metadata=md)

    out = np.copy(result).view(dt)
    result_weight = get_result_array(result, "weight")

    truncated_ell = ()
    truncated_weight = ()

    for axis, ell, w, maxval in zip(axes, ells, result_weight, ell_max):
        mask = ell <= maxval
        n = np.count_nonzero(mask)

        ell_trunc = ell[mask]
        w_trunc = w[mask]

        shape = out.shape[:axis] + (n,) + out.shape[axis + 1 :]
        tmp = np.empty(shape, dtype=dt)

        for before in np.ndindex(shape[:axis]):
            for after in np.ndindex(shape[axis + 1 :]):
                k_in = (*before, mask, *after)
                k_out = (*before, slice(None), *after)
                tmp[k_out] = out[k_in]

        out = tmp
        truncated_ell += (ell_trunc,)
        truncated_weight += (w_trunc,)

    if len(axes) == 1:
        truncated_ell = truncated_ell[0]
        truncated_weight = truncated_weight[0]

    return Result(
        out,
        spin=result.spin,
        ell=truncated_ell,
        axis=axes,
        weight=truncated_weight,
    )
