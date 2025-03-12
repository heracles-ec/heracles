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
    "get_gaussian_target",
    "get_shrinkage",
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
    "Fields2Components",
    "dict2mat",
    "mat2dict",
    "cov2spinblocks",
]

from .cls import get_cls
from .bias_corrrection import get_bias, get_delete_fsky, get_biasjk, correct_bias
from .mask_correction import compute_mask_correction, correct_mask
from .delete1 import (
    get_delete1_cov,
    get_gaussian_target,
    get_shrinkage,
    shrink_cov,
)
from .delete2 import get_delete2_cov, get_delete2_correction
from .dices import get_dices_cov
from .utils import (
    cov2corr,
    add_to_Cls,
    sub_to_Cls,
    get_W,
    Fields2Components,
    dict2mat,
    mat2dict,
    cov2spinblocks,
)
from .utils_polspice import cl2corr, corr2cl
