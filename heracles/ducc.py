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
Module for discrete spherical harmonic transforms with ducc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from heracles.core import external_dependency_explainer, update_metadata

with external_dependency_explainer:
    import ducc0

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from numpy.typing import DTypeLike, NDArray


class DiscreteMapper:
    """
    Mapper that creates alms directly.
    """

    def __init__(
        self,
        lmax: int,
        *,
        dtype: DTypeLike = np.complex128,
        nthreads: int = 0,
    ) -> None:
        """
        Mapper for alms.
        """
        self.__lmax = lmax
        self.__dtype = np.dtype(dtype)
        self.__nthreads = nthreads

    @property
    def lmax(self) -> int:
        """
        The maximum angular mode number.
        """
        return self.__lmax

    @property
    def area(self) -> float:
        """
        The effective area for this mapper.
        """
        return 1.0

    def create(
        self,
        *dims: int,
        spin: int = 0,
    ) -> NDArray[Any]:
        """
        Create zero alms.
        """
        lmax = self.__lmax
        m = np.zeros((*dims, (lmax + 1) * (lmax + 2) // 2), dtype=self.__dtype)
        update_metadata(
            m,
            geometry="discrete",
            kernel="none",
            lmax=lmax,
            spin=spin,
        )
        return m

    def map_values(
        self,
        lon: NDArray[Any],
        lat: NDArray[Any],
        data: NDArray[Any],
        values: NDArray[Any],
    ) -> None:
        """
        Add values to alms.
        """

        md: Mapping[str, Any] = data.dtype.metadata or {}

        flatten = values.ndim == 1
        if flatten:
            values = values.reshape(1, -1)

        epsilon: float
        if values.dtype == np.float64:
            epsilon = 1e-12
        elif values.dtype == np.float32:
            epsilon = 1e-5
        else:
            values = values.astype(np.float64)
            epsilon = 1e-12

        spin = md.get("spin", 0)

        loc = np.empty((lon.size, 2), dtype=np.float64)
        loc[:, 0] = np.radians(90.0 - lat)
        loc[:, 1] = np.radians(lon % 360.0)

        alms = ducc0.sht.adjoint_synthesis_general(
            map=values,
            spin=spin,
            lmax=self.__lmax,
            loc=loc,
            epsilon=epsilon,
            nthreads=self.__nthreads,
        )

        if flatten:
            alms = alms[0]

        data += alms

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        Does nothing, since inputs are alms already.
        """
        return data

    def resample(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        Change LMAX of alm.
        """
        *dims, n = data.shape
        lmax_in = (int((8 * n + 1) ** 0.5 + 0.01) - 3) // 2
        lmax_out = self.__lmax
        lmax = min(lmax_in, lmax_out)
        out = np.zeros(
            (*dims, (lmax_out + 1) * (lmax_out + 2) // 2),
            dtype=self.__dtype,
        )
        i = j = 0
        for m in range(lmax + 1):
            out[..., j : j + lmax - m + 1] = data[..., i : i + lmax - m + 1]
            i += lmax_in - m + 1
            j += lmax_out - m + 1
        return out
