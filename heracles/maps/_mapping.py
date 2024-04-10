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
"""
Module for creating maps from fields and catalogues.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import coroutines

from heracles.core import TocDict, toc_match

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from numpy.typing import NDArray

    from heracles.catalog import Catalog
    from heracles.fields import Field
    from heracles.progress import Progress, ProgressTask


async def _map_progress(
    key: tuple[Any, ...],
    field: Field,
    catalog: Catalog,
    progress: Progress | None,
) -> NDArray:
    """
    Coroutine that keeps track of progress.
    """

    task: ProgressTask | None
    if progress is not None:
        name = "[" + ", ".join(map(str, key)) + "]"
        task = progress.task(name, subtask=True, total=None)
    else:
        task = None

    result = await field(catalog, progress=task)

    if progress is not None:
        task.remove()
        progress.advance(progress.task_ids[0])

    return result


def map_catalogs(
    fields: Mapping[Any, Field],
    catalogs: Mapping[Any, Catalog],
    *,
    parallel: bool = False,
    out: MutableMapping[tuple[Any, Any], NDArray] | None = None,
    include: Sequence[tuple[Any, Any]] | None = None,
    exclude: Sequence[tuple[Any, Any]] | None = None,
    progress: bool = False,
) -> MutableMapping[tuple[Any, Any], NDArray]:
    """Make maps for a set of catalogues."""

    # the toc dict of maps
    if out is None:
        out = TocDict()

    # collect groups of items to go through
    # items are tuples of (key, field, catalog)
    groups = [
        [((i, j), field, catalog) for i, field in fields.items()]
        for j, catalog in catalogs.items()
    ]

    # flatten groups for parallel processing
    if parallel:
        groups = [sum(groups, [])]

    # display a progress bar if asked to
    progressbar: Progress | nullcontext
    if progress:
        from heracles.progress import Progress

        # create the progress bar
        # add the main task -- this must be the first task
        progressbar = Progress()
        progressbar.add_task("mapping", total=sum(map(len, groups)))
    else:
        progressbar = nullcontext()

    # process all groups of fields and catalogues
    with progressbar as prog:
        for items in groups:
            # fields return coroutines, which are ran concurrently
            keys, coros = [], []
            for key, field, catalog in items:
                if toc_match(key, include, exclude):
                    keys.append(key)
                    coros.append(_map_progress(key, field, catalog, prog))

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

        if prog is not None:
            prog.refresh()

    # return the toc dict
    return out


def transform_maps(
    fields: Mapping[Any, Field],
    maps: Mapping[tuple[Any, Any], NDArray],
    *,
    out: MutableMapping[tuple[Any, Any], NDArray] | None = None,
    progress: bool = False,
    **kwargs,
) -> MutableMapping[tuple[Any, Any], NDArray]:
    """transform a set of maps to alms"""

    # the output toc dict
    if out is None:
        out = TocDict()

    # display a progress bar if asked to
    progressbar: Progress | nullcontext
    if progress:
        from heracles.progress import Progress

        progressbar = Progress()
        task = progressbar.task("transform", total=len(maps))
    else:
        progressbar = nullcontext()

    # convert maps to alms, taking care of complex and spin-weighted maps
    with progressbar as prog:
        for (k, i), m in maps.items():
            if progress:
                subtask = prog.task(
                    f"[{k}, {i}]",
                    subtask=True,
                    start=False,
                    total=None,
                )

            try:
                field = fields[k]
            except KeyError:
                msg = f"unknown field name: {k}"
                raise ValueError(msg)

            alms = field.mapper_or_error.transform(m)

            if isinstance(alms, tuple):
                out[f"{k}_E", i] = alms[0]
                out[f"{k}_B", i] = alms[1]
            else:
                out[k, i] = alms

            del m, alms

            if progress:
                subtask.remove()
                task.update(advance=1)

        if prog is not None:
            prog.refresh()

    # return the toc dict of alms
    return out
