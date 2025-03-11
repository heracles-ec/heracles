# Heracles: Euclid code for harmonic-space statistics on the sphere
#
# Copyright (C) 2023-2025 Euclid Science Ground Segment
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
"""Module for processing of configuration."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import heracles

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, TypeGuard
    from numpy.typing import NDArray
    from heracles import Field, Mapper

    TwoPointKey = (
        tuple[()]
        | tuple[str]
        | tuple[str, str]
        | tuple[str, str, int]
        | tuple[str, str, int, int]
    )


def _is_list_of_str(obj: Any) -> TypeGuard[list[str]]:
    """Type guard for list of strings."""
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)


class FieldType(enum.Enum):
    positions: type[heracles.Field] = heracles.Positions
    shears: type[heracles.Field] = heracles.Shears
    visibility: type[heracles.Field] = heracles.Visibility
    weights: type[heracles.Field] = heracles.Weights

    def __str__(self) -> str:
        return self.name


class MapperType(enum.Enum):
    none = enum.auto()
    healpix = enum.auto()
    discrete = enum.auto()

    def __str__(self) -> str:
        return self.name


class BinSpacing(enum.Enum):
    """Spacing for binning.

    Values are functions for the spacing and its inverse.

    """

    linear = (lambda x: x, lambda x: x)
    log = (np.log, np.exp)
    log1p = (np.log1p, np.expm1)
    sqrt = (np.sqrt, np.square)


class SelectionConfig(NamedTuple):
    key: int
    selection: str
    visibility: str | None


class CatalogConfig(NamedTuple):
    source: str
    fields: list[str]
    label: str | None
    visibility: str | None
    selections: list[SelectionConfig]


class TwoPointConfig(NamedTuple):
    lmin: int | None
    lmax: int | None
    l2max: int | None
    l3max: int | None
    debias: bool
    bins: NDArray[Any] | None
    weights: str | None


class Config(NamedTuple):
    catalogs: list[CatalogConfig]
    fields: dict[str, Field]
    spectra: dict[TwoPointKey, TwoPointConfig]


def _load_selection(config: Mapping[str, Any]) -> SelectionConfig:
    """Load a selection configuration from a dictionary."""

    if missing := {"key", "selection"} - config.keys():
        raise ValueError("missing option: " + ", ".join(missing))

    if unknown := config.keys() - {"key", "selection", "visibility"}:
        raise ValueError("unknown option: " + ", ".join(unknown))

    key = config["key"]
    if not isinstance(key, int):
        raise ValueError(f"key: expected int, got {type(key)}")

    selection = config["selection"]
    if not isinstance(selection, str):
        raise ValueError(f"selection: expected str, got {type(selection)}")

    visibility = config.get("visibility")
    if visibility is not None and not isinstance(visibility, str):
        raise ValueError(f"visibility: expected str, got {type(visibility)}")

    return SelectionConfig(
        key=key,
        selection=selection,
        visibility=visibility,
    )


def _load_catalog(config: Mapping[str, Any]) -> CatalogConfig:
    """Load a catalog configuration from a dictionary."""
    # create a copy so we can pop() entries
    config = dict(config)

    source = config.pop("source")
    if not isinstance(source, str):
        raise ValueError(f"source: expected str, got {type(source)}")

    fields = config.pop("fields")
    if not _is_list_of_str(fields):
        raise ValueError("fields: expected list of str")

    label = config.pop("label", None)
    if label is not None and not isinstance(label, str):
        raise ValueError(f"label: expected str, got {type(label)}")

    visibility = config.pop("visibility", None)
    if visibility is not None and not isinstance(visibility, str):
        raise ValueError(f"visibility: expected str, got {type(visibility)}")

    _selections = config.pop("selections", [])
    if not isinstance(_selections, list):
        raise ValueError(f"selections: expected list, got {type(_selections)}")
    selections = list(map(_load_selection, _selections))

    # check unknown options
    if config:
        raise ValueError("unknown options: " + ", ".join(map(repr, config)))

    return CatalogConfig(
        fields=fields,
        source=source,
        label=label,
        selections=selections,
        visibility=visibility,
    )


def _load_mapper(config: dict[str, Any]) -> Mapper | None:
    """Load a mapper from a dictionary.

    For simplicity, this consumes keys from the dictionary.

    """
    _mapper = config.pop("mapper", "healpix")
    if not isinstance(_mapper, str):
        raise ValueError(f"mapper: expected str, got {type(_mapper)}")
    try:
        mapper = MapperType[_mapper]
    except KeyError:
        raise ValueError(f"mapper: invalid value: {_mapper!r}") from None

    nside = config.pop("nside", None)
    if nside is not None and not isinstance(nside, int):
        raise ValueError(f"nside: expected int, got {type(nside)}")

    lmax = config.pop("lmax", None)
    if lmax is not None and not isinstance(lmax, int):
        raise ValueError(f"lmax: expected int, got {type(lmax)}")

    if mapper is MapperType.healpix:
        from heracles.healpy import HealpixMapper

        if nside is None:
            raise ValueError("missing option: nside")

        return HealpixMapper(nside, lmax)

    if mapper is MapperType.discrete:
        from heracles.ducc import DiscreteMapper

        if lmax is None:
            raise ValueError("missing option: nside")

        return DiscreteMapper(lmax)

    # mapper is "none" here
    return None


def _load_field(config: Mapping[str, Any]) -> tuple[str, Field]:
    """Load a field from a dictionary."""
    # make a copy so that we can pop items
    config = dict(config)

    key = config.pop("key")
    if not isinstance(key, str):
        raise ValueError(f"key: expected str, got {type(key)}")

    _cls = config.pop("type")
    if not isinstance(_cls, str):
        raise ValueError(f"{key}: type: expected str, got {type(_cls)}")
    try:
        cls = FieldType[_cls].value
    except KeyError:
        raise ValueError(f"{key}: type: invalid value: {_cls!r}") from None

    columns = config.pop("columns", [])
    if not _is_list_of_str(columns):
        raise ValueError("{key}: columns: expected list of str")

    mask = config.pop("mask", None)
    if mask is not None and not isinstance(mask, str):
        raise ValueError(f"{key}: mask: expected str, got {type(mask)}")

    try:
        mapper = _load_mapper(config)
    except ValueError as exc:
        raise ValueError(f"{key}: {exc!s}") from None

    # check unknown options
    if config:
        raise ValueError(f"{key}: unknown options: " + ", ".join(map(repr, config)))

    return key, cls(mapper, *columns, mask=mask)


def _load_bins(
    config: dict[str, Any],
    lmin: int,
    lmax: int,
) -> tuple[NDArray[Any], str | None]:
    """Construct angular bins from config."""
    config = dict(config)

    n = config.pop("n")
    if not isinstance(n, int):
        raise ValueError(f"n: expected int, got {type(n)}")
    if n < 2:
        raise ValueError(f"n: invalid number of bins: {n}")

    _spacing = config.pop("spacing", "linear")
    if not isinstance(_spacing, str):
        raise ValueError(f"spacing: expected str, got {type(_spacing)}")
    try:
        op, inv = BinSpacing[_spacing].value
    except KeyError:
        raise ValueError(f"spacing: invalid value: {_spacing!r}") from None

    weights = config.pop("weights", None)
    if weights is not None and not isinstance(weights, str):
        raise ValueError(f"weights: expected str, got {type(weights)}")

    bins = inv(np.linspace(op(lmin), op(lmax + 1), n + 1))
    # fix first and last array element to be exact
    bins[0], bins[-1] = lmin, lmax + 1

    return bins, weights


def _load_twopoint(config: Mapping[str, Any]) -> tuple[TwoPointKey, TwoPointConfig]:
    """Load two-point config from a dictionary."""
    # make a copy so that we can pop items
    config = dict(config)

    _key = config.pop("key")
    if not isinstance(_key, (str, list)):
        raise ValueError(f"key: expected str or list, got {type(_key)}")
    if isinstance(_key, str):
        key = heracles.key_from_str(_key)
    else:
        key = tuple(_key)
        # stringified version for errors
        _key = heracles.key_to_str(key)

    lmin = config.pop("lmin", None)
    if lmin is not None and not isinstance(lmin, int):
        raise ValueError(f"{_key}: lmin: expected int, got {type(lmin)}")

    lmax = config.pop("lmax", None)
    if lmax is not None and not isinstance(lmax, int):
        raise ValueError(f"{_key}: lmax: expected int, got {type(lmax)}")

    l2max = config.pop("l2max", None)
    if l2max is not None and not isinstance(l2max, int):
        raise ValueError(f"{_key}: l2max: expected int, got {type(l2max)}")

    l3max = config.pop("l3max", None)
    if l3max is not None and not isinstance(l3max, int):
        raise ValueError(f"{_key}: l3max: expected int, got {type(l3max)}")

    debias = config.pop("debias", True)
    if not isinstance(debias, bool):
        raise ValueError(f"{_key}: debias: expected bool, got {type(debias)}")

    _bins = config.pop("bins", None)
    if _bins is not None:
        if not isinstance(_bins, dict):
            raise ValueError(f"{_key}: bins: expected dict, got {type(_bins)}")
        if lmin is None or lmax is None:
            raise ValueError(f"{_key}: bins: angular binning requires lmin and lmax")
        try:
            bins, weights = _load_bins(_bins, lmin, lmax)
        except ValueError as exc:
            raise ValueError(f"{_key}: bins: {exc!s}")
    else:
        bins, weights = None, None

    # check unknown options
    if config:
        raise ValueError(f"{_key}: unknown options: " + ", ".join(map(repr, config)))

    return key, TwoPointConfig(
        lmin=lmin,
        lmax=lmax,
        l2max=l2max,
        l3max=l3max,
        debias=debias,
        bins=bins,
        weights=weights,
    )


def load(config: Mapping[str, Any]) -> Config:
    """Load configuration from a dictionary."""
    # make a copy so that we can pop entries
    config = dict(config)

    _catalogs = config.pop("catalogs", [])
    if not isinstance(_catalogs, list):
        raise ValueError(f"catalogs: expected list, got {type(_catalogs)}")
    catalogs: list[CatalogConfig] = []
    for i, item in enumerate(_catalogs):
        try:
            catalog = _load_catalog(item)
        except ValueError as exc:
            raise ValueError(f"catalogs[{i + 1}]: {exc!s}") from None
        else:
            catalogs.append(catalog)

    _fields = config.pop("fields", [])
    if not isinstance(_fields, list):
        raise ValueError(f"fields: expected list, got {type(_fields)}")
    fields: dict[str, Field] = {}
    for i, item in enumerate(_fields):
        try:
            key, field = _load_field(item)
        except ValueError as exc:
            raise ValueError(f"fields[{i + 1}]: {exc!s}") from None
        else:
            fields[key] = field

    _spectra = config.pop("spectra", [])
    if not isinstance(_spectra, list):
        raise ValueError(f"spectra: expected list, got {type(_spectra)}")
    spectra: dict[TwoPointKey, TwoPointConfig] = {}
    for i, item in enumerate(_spectra):
        try:
            key2, twopoint = _load_twopoint(item)
        except ValueError as exc:
            raise ValueError(f"spectra[{i + 1}]: {exc!s}") from None
        else:
            spectra[key2] = twopoint

    return Config(
        catalogs=catalogs,
        fields=fields,
        spectra=spectra,
    )
