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
    # cls
    "jackknife_cls",
    # bias_correction
    "jackknife_fsky",
    "jackknife_bias",
    "correct_bias",
    # mask_correction
    "correct_mask",
    # Delete1
    "jackknife_covariance",
    # Delete2
    "debias_covariance",
    "delete2_correction",
    # PolSpice
    "cl2corr",
    "corr2cl",
]

from .cls import (
    jackknife_cls,
)
from .bias_correction import (
    jackknife_fsky,
    jackknife_bias,
    correct_bias,
)
from .mask_correction import (
    correct_mask,
)
from .delete1 import (
    jackknife_covariance,
)
from .delete2 import (
    debias_covariance,
    delete2_correction,
)
from .polspice import (
    cl2corr,
    corr2cl,
)
