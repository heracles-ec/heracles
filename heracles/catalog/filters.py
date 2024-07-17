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
"""module for catalogue filters"""

import warnings

import numpy as np


class InvalidValueFilter:
    """Filter invalid values from a catalogue."""

    def __init__(self, *columns, weight=None, warn=True):
        """Filter invalid values in the given columns.

        If ``warn`` is true, invalid values will emit a warning.

        """

        self.columns = columns
        self.weight = weight
        self.warn = warn

    def __repr__(self):
        name = self.__class__.__name__
        args = list(map(repr, self.columns))
        args += [f"weight={self.weight!r}", f"warn={self.warn!r}"]
        args = ", ".join(args)
        return f"{name}({args})"

    def __call__(self, page):
        """Filter a catalog page."""

        invalid_mask = np.zeros(page.size, dtype=bool)
        for col in self.columns:
            invalid_mask |= np.isnan(page[col])
        if self.weight is not None:
            invalid_mask &= page[self.weight] != 0
        invalid = np.where(invalid_mask)[0]
        if len(invalid) > 0:
            if self.warn:
                warnings.warn("WARNING: catalog contains invalid values")
            page.delete(invalid)


class FootprintFilter:
    """Filter a catalogue using a footprint map."""

    def __init__(self, footprint, lon, lat):
        """Filter using the given footprint map and position columns."""
        from healpy import get_nside

        self._footprint = footprint
        self._nside = get_nside(footprint)
        self._lonlat = (lon, lat)

    @property
    def footprint(self):
        """footprint for filter"""
        return self._footprint

    @property
    def lonlat(self):
        """longitude and latitude columns"""
        return self._lonlat

    def __repr__(self):
        name = self.__class__.__name__
        lon, lat = self.lonlat
        return f"{name}(..., {lon!r}, {lat!r})"

    def __call__(self, page):
        """filter catalog page"""

        from healpy import ang2pix

        lon, lat = self._lonlat
        ipix = ang2pix(self._nside, page[lon], page[lat], lonlat=True)
        exclude = np.where(self._footprint[ipix] == 0)[0]
        page.delete(exclude)
