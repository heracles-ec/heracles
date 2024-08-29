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
Module for the integration with the rich package.
Contains a progress bar implementation.
"""

from __future__ import annotations

import rich.box
import rich.panel
import rich.progress


class _RichProgressBar(rich.progress.Progress):
    """
    Rich progress bar subclass with customisations.
    """

    @classmethod
    def get_default_columns(cls):
        """
        Default columns for progress reporting.
        """
        return (
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(bar_width=20),
            rich.progress.TaskProgressColumn(),
            rich.progress.TimeElapsedColumn(),
        )

    def make_tasks_table(self, tasks):
        """
        Create a table of tasks sorted by their depths.
        """
        sorted_tasks = sorted(tasks, key=lambda task: int(task.fields.get("depth", -1)))
        table = super().make_tasks_table(sorted_tasks)
        table.box = rich.box.HORIZONTALS
        depth = -1
        for i, task in enumerate(sorted_tasks):
            if (d := task.fields.get("depth", -1)) != depth:
                depth = d
                if i > 0:
                    table.rows[i - 1].end_section = True
        return table


class Progress:
    """
    Progress bar using rich.
    """

    def __init__(
        self,
        label: str,
        *,
        progress: rich.progress.Progress | None = None,
        depth: int = 0,
    ) -> None:
        if progress is None:
            self.progress = _RichProgressBar()
        else:
            self.progress = progress
        self.label = label
        self.depth = depth
        self.task_id: rich.progress.TaskID | None = None

    def __enter__(self) -> "Progress":
        if not self.progress.tasks:
            self.progress.start()
        if self.task_id is None:
            self.task_id = self.progress.add_task(
                self.label,
                start=True,
                total=None,
                depth=self.depth,
            )
        else:
            self.progress.start_task(self.task_id)
        self.progress.refresh()
        return self

    def __exit__(self, *exc) -> None:
        if self.task_id is not None:
            self.progress.stop_task(self.task_id)
            self.progress.remove_task(self.task_id)
            self.task_id = None
        if not self.progress.tasks:
            self.progress.stop()
        self.progress.refresh()

    def update(self, current: int | None = None, total: int | None = None):
        if self.task_id is not None:
            self.progress.update(self.task_id, total=total, completed=current)
        self.progress.refresh()

    def task(self, label: str) -> "Progress":
        return self.__class__(label, progress=self.progress, depth=self.depth + 1)
