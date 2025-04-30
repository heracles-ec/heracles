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
    "jackknife_covariance",
    "debias_covariance",
    "delete2_correction",
    # mask_correction
    "correct_mask",
    "cl2corr",
    "corr2cl",
    # shrinkage
    "shrink_covariance",
    "shrinkage_factor",
    "gaussian_covariance",
    # io
    "fields2componentscomponents2data",
    "flatten",
]

from .jackknife import (
    jackknife_cls,
    jackknife_fsky,
    jackknife_bias,
    correct_bias,
    jackknife_covariance,
    debias_covariance,
    delete2_correction,
)
from .mask_correction import (
    correct_mask,
    cl2corr,
    corr2cl,
)
from .shrinkage import (
    shrink_covariance,
    shrinkage_factor,
    gaussian_covariance,
)
from .io import (
    flatten,
)
