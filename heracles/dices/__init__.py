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
    "jackknife",
    "get_cls",
    # bias_correction
    "get_bias",
    "get_delete_fsky",
    "get_biasjk",
    "correct_bias",
    # mask_correction
    "compute_mask_correction",
    "correct_mask",
    # Delete1
    "get_delete1_cov",
    "get_gaussian_cov",
    "get_shrinkage_factor",
    "shrink_cov",
    # Delete2
    "get_delete2_cov",
    "get_delete2_correction",
    # DICES
    "get_dices_cov",
    # PolSpice
    "cl2corr",
    "corr2cl",
    # utils
    "cov2corr",
    "add_to_Cls",
    "sub_to_Cls",
    "get_W",
    "get_Wbar",
    "Fields2Components",
    # io
    "Fields2Components",
    "Components2Data",
    "Data2Components",
    "Components2Fields",
]

from .cls import jackknife, get_cls
from .bias_correction import (
    get_bias,
    get_delete_fsky,
    get_biasjk,
    correct_bias,
)
from .mask_correction import compute_mask_correction, correct_mask
from .delete1 import (
    get_delete1_cov,
    get_gaussian_cov,
    get_shrinkage_factor,
    shrink_cov,
)
from .delete2 import get_delete2_cov, get_delete2_correction
from .dices import get_dices_cov
from .utils import (
    cov2corr,
    add_to_Cls,
    sub_to_Cls,
    get_W,
    get_Wbar,
)
from .io import (
    Fields2Components,
    Components2Data,
    Data2Components,
    Components2Fields,
)
from .polspice import cl2corr, corr2cl
