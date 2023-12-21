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
"""
Module for HEALPix maps.
"""

from __future__ import annotations

from functools import cached_property, wraps
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from numba import njit

from heracles.core import update_metadata

from ._mapper import Mapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, DTypeLike


def _asnative(arr):
    """
    Return *arr* in native byte order.
    """
    if arr.dtype.byteorder != "=":
        return arr.newbyteorder("=").byteswap()
    return arr


def _nativebyteorder(fn):
    """utility decorator to convert inputs to native byteorder"""

    @wraps(fn)
    def wrapper(*args):
        native = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                native.append(tuple(map(_asnative, arg)))
            else:
                native.append(_asnative(arg))
        return fn(*native)

    return wrapper


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map0(ipix, maps):
    """
    Compiled function to map positions.
    """
    n = len(maps)
    for i in ipix:
        for k in range(n):
            maps[k][i] += 1


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map0w(ipix, maps, weight):
    """
    Compiled function to map positions with weights.
    """
    n = len(maps)
    for j, i in enumerate(ipix):
        w = weight[j]
        for k in range(n):
            maps[k][i] += w


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _map(ipix, maps, values):
    """
    Compiled function to map values.
    """
    n = len(maps)
    for j, i in enumerate(ipix):
        for k in range(n):
            maps[k][i] += values[k][j]


@_nativebyteorder
@njit(nogil=True, fastmath=True)
def _mapw(ipix, maps, values, weight):
    """
    Compiled function to map values with weights.
    """
    n = len(maps)
    for j, i in enumerate(ipix):
        w = weight[j]
        for k in range(n):
            maps[k][i] += w * values[k][j]


class Healpix(Mapper, kernel="healpix"):
    """
    Mapper for HEALPix maps.  HEALPix maps have a resolution parameter,
    available as the *nside* property.
    """

    def __init__(
        self,
        nside: int,
        dtype: DTypeLike = np.float64,
        *,
        datapath: str | None = None,
    ) -> None:
        """
        Mapper for HEALPix maps with the given *nside* parameter.
        """
        super().__init__()
        self.__nside = nside
        self.__npix = hp.nside2npix(nside)
        self.__dtype = np.dtype(dtype)
        self.__datapath = datapath
        self._metadata |= {
            "nside": nside,
        }

    @property
    def nside(self) -> int:
        """
        The resolution parameter of the HEALPix map.
        """
        return self.__nside

    @property
    def dtype(self) -> DTypeLike:
        """
        Data type for HEALPix maps.
        """
        return self.__dtype

    @property
    def size(self) -> int:
        """
        Size of HEALPix maps for this *nside* parameter.
        """
        return self.__npix

    @cached_property
    def area(self) -> float:
        """
        The HEALPix pixel area for this mapper.
        """
        return hp.nside2pixarea(self.__nside)

    def map_values(
        self,
        lon: ArrayLike,
        lat: ArrayLike,
        maps: Sequence[ArrayLike],
        values: Sequence[ArrayLike] | None = None,
        weight: ArrayLike | None = None,
    ) -> None:
        """
        Add values to HEALPix maps.
        """

        # pixel indices of given positions
        ipix = hp.ang2pix(self.__nside, lon, lat, lonlat=True)

        # sum weighted values in each pixel
        # only use what is available, to minimise number of array operations
        # turn sequences into tuples for numba interoperability
        if values is None:
            if weight is None:
                _map0(ipix, maps)
            else:
                _map0w(ipix, maps, weight)
        else:
            if weight is None:
                _map(ipix, maps, values)
            else:
                _mapw(ipix, maps, values, weight)

    def transform(
        self,
        maps: ArrayLike,
        lmax: int | None = None,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """
        Spherical harmonic transform of HEALPix maps.
        """

        md = maps.dtype.metadata or {}
        spin = md.get("spin", 0)

        if spin == 0:
            pol = False
        elif spin == 2:
            maps = [np.zeros(self.__npix), maps[0], maps[1]]
            pol = True
        else:
            msg = f"spin-{spin} maps not yet supported"
            raise NotImplementedError(msg)

        alms = hp.map2alm(
            maps,
            lmax=lmax,
            pol=pol,
            use_pixel_weights=True,
            datapath=self.__datapath,
        )

        if spin == 0:
            update_metadata(alms, **md)
        else:
            alms = (alms[1], alms[2])
            update_metadata(alms[0], **md)
            update_metadata(alms[1], **md)

        return alms
