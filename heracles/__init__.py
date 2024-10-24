# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2024 Euclid Science Ground Segment
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
Main module of the *Heracles* package.
"""

__all__ = [
    # _version
    "__version__",
    "__version_tuple__",
    # catalog
    "ArrayCatalog",
    "Catalog",
    "CatalogBase",
    "CatalogPage",
    "FitsCatalog",
    "FootprintFilter",
    "InvalidValueFilter",
    # core
    "TocDict",
    "toc_filter",
    "toc_match",
    "update_metadata",
    # fields
    "ComplexField",
    "Field",
    "Positions",
    "ScalarField",
    "Shears",
    "Spin2Field",
    "Visibility",
    "Weights",
    # io
    "read_vmap",
    "read_alms",
    "read_cls",
    "read_cov",
    "read_maps",
    "read_mms",
    "write_alms",
    "write_cls",
    "write_cov",
    "write_maps",
    "write_mms",
    # mapper
    "Mapper",
    # mapping
    "map_catalogs",
    "transform",
    # progress
    "NoProgress",
    "Progress",
    # twopoint
    "angular_power_spectra",
    "debias_cls",
    "mixing_matrices",
    "bin2pt",
    "binned_cls",
    "binned_mms",
]

try:
    from ._version import __version__, __version_tuple__
except ModuleNotFoundError:
    __version__ = None
    __version_tuple__ = None

from .catalog import (
    ArrayCatalog,
    Catalog,
    CatalogBase,
    CatalogPage,
    FitsCatalog,
    FootprintFilter,
    InvalidValueFilter,
)

from .core import (
    TocDict,
    toc_filter,
    toc_match,
    update_metadata,
)

from .fields import (
    ComplexField,
    Field,
    Positions,
    ScalarField,
    Shears,
    Spin2Field,
    Visibility,
    Weights,
)

from .io import (
    read_vmap,
    read_alms,
    read_cls,
    read_cov,
    read_maps,
    read_mms,
    write_alms,
    write_cls,
    write_cov,
    write_maps,
    write_mms,
)

from .mapper import (
    Mapper,
)

from .mapping import (
    map_catalogs,
    transform,
)

from .progress import (
    NoProgress,
    Progress,
)

from .twopoint import (
    angular_power_spectra,
    debias_cls,
    mixing_matrices,
    bin2pt,
    binned_cls,
    binned_mms,
)
