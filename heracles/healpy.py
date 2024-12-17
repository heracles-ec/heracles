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
Module for HEALPix maps.
"""

from __future__ import annotations

from functools import cached_property, wraps
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from heracles.core import external_dependency_explainer, update_metadata

with external_dependency_explainer:
    import healpy as hp

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from numpy.typing import DTypeLike, NDArray


def _nativebyteorder(fn):
    """utility decorator to convert inputs to native byteorder"""

    @wraps(fn)
    def wrapper(*args):
        newargs = []
        for arr in args:
            if arr.dtype.byteorder != "=":
                arr = arr.view(arr.dtype.newbyteorder('=')).byteswap()
            newargs.append(arr)
        return fn(*newargs)

    return wrapper


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map(ipix, maps, values):
    """
    Compiled function to map values.
    """
    for j, i in enumerate(ipix):
        maps[..., i] += values[..., j]


class HealpixMapper:
    """
    Mapper for HEALPix maps.
    """

    DATAPATH: str | None = None

    def __init__(
        self,
        nside: int,
        lmax: int | None = None,
        *,
        deconvolve: bool | None = None,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """
        Mapper for HEALPix maps.
        """
        if lmax is None:
            lmax = 3 * nside // 2
        if deconvolve is None:
            deconvolve = True
        super().__init__()
        self.__nside = nside
        self.__lmax = lmax
        self.__deconv = deconvolve
        self.__dtype = np.dtype(dtype)

    @property
    def nside(self) -> int:
        """
        The resolution parameter of the HEALPix map.
        """
        return self.__nside

    @property
    def lmax(self) -> int:
        """
        The lmax parameter of the HEALPix map.
        """
        return self.__lmax

    @property
    def deconvolve(self) -> bool:
        """
        Whether or not the HEALPix pixel window function is deconvolved.
        """
        return self.__deconv

    @cached_property
    def area(self) -> float:
        """
        The HEALPix pixel area for this mapper.
        """
        return hp.nside2pixarea(self.__nside)

    def create(
        self,
        *dims: int,
        spin: int = 0,
    ) -> NDArray[Any]:
        """
        Create a new HEALPix map.
        """
        m = np.zeros((*dims, hp.nside2npix(self.__nside)), dtype=self.__dtype)
        update_metadata(
            m,
            geometry="healpix",
            kernel="healpix",
            nside=self.__nside,
            lmax=self.__lmax,
            deconv=self.__deconv,
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
        Add values to HEALPix maps.
        """

        # pixel indices of given positions
        ipix = hp.ang2pix(self.__nside, lon, lat, lonlat=True)

        # sum values in each pixel
        _map(ipix, data, values)

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        Spherical harmonic transform of HEALPix maps.
        """

        md: Mapping[str, Any] = data.dtype.metadata or {}
        spin = md.get("spin", 0)
        pw: NDArray[Any] | None = None

        if spin == 0:
            pol = False
            if self.__deconv:
                pw = hp.pixwin(self.__nside, lmax=self.__lmax, pol=False)
        elif spin == 2:
            data = np.r_[self.create(1), data]
            pol = True
            if self.__deconv:
                pw = hp.pixwin(self.__nside, lmax=self.__lmax, pol=True)[1]
        else:
            msg = f"spin-{spin} maps not yet supported"
            raise NotImplementedError(msg)

        alm = hp.map2alm(
            data,
            lmax=self.__lmax,
            pol=pol,
            use_pixel_weights=True,
            datapath=self.DATAPATH,
        )

        if pw is not None:
            fl = np.ones(self.__lmax + 1)
            fl[abs(spin) :] /= pw[abs(spin) :]
            for i in np.ndindex(*alm.shape[:-1]):
                alm[i] = hp.almxfl(alm[i], fl)
            del fl

        if spin != 0:
            alm = alm[1:].copy()

        update_metadata(alm, **md)

        return alm

    def resample(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        Change resolution of HEALPix map.
        """
        return hp.ud_grade(data, self.__nside, dtype=self.__dtype)
