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
Module for creating maps from fields and catalogues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import coroutines

from heracles.core import TocDict, toc_match
from heracles.progress import Progress, NoProgress

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from numpy.typing import NDArray

    from heracles.catalog import Catalog
    from heracles.fields import Field


async def _map_field(
    key: tuple[Any, ...],
    field: Field,
    catalog: Catalog,
    progress: Progress,
    task_done: Callable[[], None],
) -> NDArray:
    """
    Coroutine to map an individual field.
    """

    label = "(" + ", ".join(map(str, key)) + ")"
    with progress.task(label) as task:
        result = await field(catalog, progress=task)

    task_done()

    return result


def map_catalogs(
    fields: Mapping[Any, Field],
    catalogs: Mapping[Any, Catalog],
    *,
    parallel: bool = False,
    out: MutableMapping[tuple[Any, Any], NDArray] | None = None,
    include: Sequence[tuple[Any, Any]] | None = None,
    exclude: Sequence[tuple[Any, Any]] | None = None,
    progress: Progress | None = None,
) -> MutableMapping[tuple[Any, Any], NDArray]:
    """Map a set of catalogues to fields."""

    # the toc dict of results
    if out is None:
        out = TocDict()

    # create dummy progress object if none was given
    if progress is None:
        progress = NoProgress()

    # collect groups of items to go through
    # items are tuples of (key, field, catalog)
    groups = [
        [((i, j), field, catalog) for i, field in fields.items()]
        for j, catalog in catalogs.items()
    ]

    # flatten groups for parallel processing
    if parallel:
        groups = [sum(groups, [])]

    # progress tracking
    current, total = 0, sum(map(len, groups))
    progress.update(0, total)

    def _task_done():
        """callback for async execution"""
        nonlocal current
        current += 1
        progress.update(current, total)

    # process all groups of fields and catalogues
    for items in groups:
        # fields return coroutines, which are ran concurrently
        keys, coros = [], []
        for key, field, catalog in items:
            if toc_match(key, include, exclude):
                keys.append(key)
                coros.append(_map_field(key, field, catalog, progress, _task_done))

        # run all coroutines concurrently
        try:
            results = coroutines.run(coroutines.gather(*coros))
        finally:
            # force-close coroutines to prevent "never awaited" warnings
            for coro in coros:
                coro.close()

        # store results
        for key, value in zip(keys, results):
            out[key] = value

        # free up memory for next group
        del results

    # return the toc dict
    return out


def transform(
    fields: Mapping[Any, Field],
    data: Mapping[tuple[Any, Any], NDArray],
    *,
    out: MutableMapping[tuple[Any, Any], NDArray] | None = None,
    progress: Progress | None = None,
) -> MutableMapping[tuple[Any, Any], NDArray]:
    """transform data to alms"""

    # the output toc dict
    if out is None:
        out = TocDict()

    # create dummy progress object if none was given
    if progress is None:
        progress = NoProgress()

    # progress reporting
    current, total = 0, len(data)

    # convert data to alms, taking care of complex and spin-weighted fields
    for (k, i), m in data.items():
        current += 1
        progress.update(current, total)

        with progress.task(f"({k}, {i})"):
            try:
                field = fields[k]
            except KeyError:
                msg = f"unknown field name: {k}"
                raise ValueError(msg) from None

            out[k, i] = field.mapper_or_error.transform(m)

    # return the toc dict of alms
    return out
