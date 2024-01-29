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
    from typing import Self

    from numpy.typing import ArrayLike, DTypeLike, NDArray

# dictionary of kernel names and their corresponding Mapper classes
_KERNELS: dict[str, type[Mapper]] = {}


def get_kernels() -> Mapping[str, type[Mapper]]:
    """
    Return a mapping of known kernel names and the classes that
    implement them.
    """
    return MappingProxyType(_KERNELS)


def mapper_from_dict(d: Mapping[str, Any]) -> Mapper:
    """
    Return a mapper that matches the given metadata.
    """

    try:
        kernel = d["kernel"]
    except KeyError:
        msg = "no 'kernel' in mapping"
        raise ValueError(msg) from None

    try:
        cls = _KERNELS[kernel]
    except KeyError:
        msg = f"unknown kernel: {kernel}"
        raise ValueError(msg) from None

    return cls.from_dict(d)


class Mapper(metaclass=ABCMeta):
    """
    Abstract base class for mappers.
    """

    __kernel: str

    def __init_subclass__(cls, /, kernel: str, **kwargs) -> None:
        """
        Initialise mapper subclasses with a *kernel* parameter.
        """
        super().__init_subclass__(**kwargs)
        cls.__kernel = kernel
        _KERNELS[kernel] = cls

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        """
        Create a new mapper instance from a dictionary of parameters.
        """

    def __init__(self) -> None:
        """
        Initialise a new mapper instance.
        """
        self._metadata: dict[str, Any] = {
            "kernel": self.__kernel,
        }

    @property
    def kernel(self) -> str:
        """
        Return the name of the kernel for this mapper.
        """
        return self.__kernel

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
    def kl(self, lmax: int, spin: int = 0) -> NDArray[Any]:
        """
        Return the convolution kernel in harmonic space.
        """

    def bl(self, lmax: int, spin: int = 0) -> None | NDArray[Any]:
        """
        Return the biasing kernel in harmonic space.
        """
        return None
