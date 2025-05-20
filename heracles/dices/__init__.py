# DICES: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
#
# This file is part of DICES.
#
# DICES is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DICES is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with DICES. If not, see <https://www.gnu.org/licenses/>.
"""
Main module of the *DICES* package.
"""

__all__ = [
    # jackknife
    "jackknife_cls",
    "jackknife_fsky",
    "jackknife_bias",
    "correct_bias",
    "jackknife_maps",
    "mask_correction",
    "jackknife_covariance",
    "debias_covariance",
    "delete2_correction",
    # shrinkage
    "shrink",
    "shrinkage_factor",
    "gaussian_covariance",
    # io
    "flatten",
    # utils
    "impose_correlation",
]

from .jackknife import (
    jackknife_cls,
    jackknife_maps,
    jackknife_fsky,
    jackknife_bias,
    correct_bias,
    mask_correction,
    jackknife_covariance,
    debias_covariance,
    delete2_correction,
)

from .shrinkage import (
    shrink,
    shrinkage_factor,
    gaussian_covariance,
)
from .io import (
    flatten,
)
from .utils import (
    impose_correlation,
)
