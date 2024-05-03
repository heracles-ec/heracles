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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray


@runtime_checkable
class Mapper(Protocol):
    """
    Protocol for mappers.
    """

    @property
    def area(self) -> float:
        """
        Effective area in steradians of one "pixel" of this mapper.
        """

    def create(
        self,
        *dims: int,
        spin: int = 0,
    ) -> NDArray[Any]:
        """
        Create a new map for this mapper.
        """

    def map_values(
        self,
        lon: NDArray[Any],
        lat: NDArray[Any],
        data: NDArray[Any],
        values: NDArray[Any],
    ) -> None:
        """
        Add values to data.
        """

    def transform(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        The spherical harmonic transform for this mapper.
        """

    def resample(self, data: NDArray[Any]) -> NDArray[Any]:
        """
        Change resolution of data, which must be in this mapper's format.
        """
