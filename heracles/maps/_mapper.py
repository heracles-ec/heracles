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
Module for the mapper base class.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike, DTypeLike

# dictionary of kernel names and their corresponding Mapper classes
_KERNELS: dict[str, type[Mapper]] = {}


def get_kernels() -> Mapping[str, type[Mapper]]:
    """
    Return a mapping of known kernel names and the classes that
    implement them.
    """
    return MappingProxyType(_KERNELS)


class MapperMeta(ABCMeta):
    """
    Metaclass for mappers.
    """

    _kernel: str | None = None

    @property
    def kernel(cls) -> str | None:
        return cls._kernel


class Mapper(metaclass=MapperMeta):
    """
    Abstract base class for mappers.
    """

    def __init_subclass__(cls, /, kernel: str, **kwargs):
        """
        Initialise mapper subclasses with a *kernel* parameter.
        """
        super().__init_subclass__(**kwargs)
        cls._kernel = kernel
        _KERNELS[kernel] = cls

    def __init__(self) -> None:
        """
        Initialise a new mapper instance.
        """
        self._metadata: dict[str, Any] = {
            "kernel": self.__class__.kernel,
        }

    @property
    def metadata(self) -> Mapping[str, Any]:
        """
        Return the metadata associated with this mapper.
        """
        return MappingProxyType(self._metadata)

    @property
    @abstractmethod
    def dtype(self) -> DTypeLike:
        """
        Data type of arrays for this mapper.
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Size of arrays for this mapper.
        """

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Area in steradians of one "pixel" of this mapper.
        """

    @abstractmethod
    def map_values(
        self,
        lon: ArrayLike,
        lat: ArrayLike,
        maps: Sequence[ArrayLike],
        values: Sequence[ArrayLike] | None = None,
        weight: ArrayLike | None = None,
    ) -> None:
        """
        Add values to maps.
        """

    @abstractmethod
    def transform(
        self,
        maps: ArrayLike,
        lmax: int | None = None,
    ) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """
        The spherical harmonic transform for this mapper.
        """

    @abstractmethod
    def deconvolve(self, alm: ArrayLike, *, inplace: bool = False) -> ArrayLike:
        """
        Remove this mapper's convolution kernel from *alm*.
        """
