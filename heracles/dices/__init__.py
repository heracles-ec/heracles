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
    # dices
    "DICES",
    # utils_cl
    "get_lgrid",
    "get_Cls_bias",
    "compsep_Cls",
    "add_to_Cls",
    "sub_to_Cls",
    "get_covkeys",
    "update_metadata",
    "cov_zero",
    "cov_add",
    "update_covariance",
    "cov2corr",
    "make_posdef",
    "get_Cl_cov",
    # SH
    "get_Clkey",
    "get_W",
    "get_T_rbar",
    "get_T_new",
    "get_covSS",
    "get_f",
    "get_lambda_star_single_rbar",
    # PolSpice
    "cl2corr",
    "corr2cl",
]

from .dices import (
    DICES,
)

from .utils_cl import (
    get_lgrid,
    get_Cls_bias,
    add_to_Cls,
    compsep_Cls,
    sub_to_Cls,
    get_covkeys,
    update_metadata,
    cov_zero,
    cov_add,
    update_covariance,
    cov2corr,
    make_posdef,
    get_Cl_cov,
)

from .utils_sh import (
    get_Clkey,
    get_W,
    get_T_rbar,
    get_T_new,
    get_covSS,
    get_f,
    get_lambda_star_single_rbar,
)

from .utils_polspice import (
    cl2corr,
    corr2cl,
)
