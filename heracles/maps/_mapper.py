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
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import ArrayLike, DTypeLike, NDArray


@runtime_checkable
class Mapper(Protocol):
    """
    Protocol for mappers.
    """

    @property
    def area(self) -> float:
        """
        Area in steradians of one "pixel" of this mapper.
        """

    def create(
        self,
        *dims: int,
        dtype: DTypeLike | None = None,
        spin: int = 0,
    ) -> NDArray[Any]:
        """
        Create a new map for this mapper.
        """

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

    def transform(self, maps: ArrayLike) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """
        The spherical harmonic transform for this mapper.
        """
