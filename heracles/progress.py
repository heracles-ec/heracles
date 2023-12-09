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
"""module for progress reporting with rich"""

try:
    import rich.box
    import rich.panel
    import rich.progress
except ModuleNotFoundError as exc:
    try:
        exc.add_note("You do not have the 'rich' package installed.")
        exc.add_note("Disabling progress reports should fix this error.")
    except AttributeError:
        pass
    raise


class ProgressTask:
    """
    A wrapper for tasks that forwards calls with their task ID.
    """

    def __init__(
        self,
        progress: rich.progress.Progress,
        task_id: rich.progress.TaskID,
    ) -> None:
        self.progress = progress
        self.task_id = task_id

    def start(self) -> None:
        self.progress.start_task(self.task_id)

    def stop(self) -> None:
        self.progress.stop_task(self.task_id)

    def remove(self) -> None:
        self.progress.remove_task(self.task_id)

    def update(self, *args, **kwargs):
        self.progress.update(self.task_id, *args, **kwargs)

    def track(self, *args, **kwargs):
        return self.progress.track(*args, task_id=self.task_id, **kwargs)


class Progress(rich.progress.Progress):
    """
    A progress bar that

    a) returns ProgressTask instances on task creation,
    b) allows creation of "subtasks" by passing subtask=True.

    Subtasks are shown below the main tasks and separated by a divider.
    """

    def task(self, *args, **kwargs) -> ProgressTask:
        task_id = self.add_task(*args, **kwargs)
        return ProgressTask(self, task_id)

    @classmethod
    def get_default_columns(cls):
        return (
            rich.progress.TextColumn("[progress.description]{task.description}"),
            rich.progress.BarColumn(bar_width=20),
            rich.progress.TaskProgressColumn(),
            rich.progress.TimeElapsedColumn(),
        )

    def make_tasks_table(self, tasks):
        def _is_subtask(task):
            return bool(task.fields.get("subtask"))

        subtask_count = sum(map(_is_subtask, tasks))
        sorted_tasks = sorted(tasks, key=_is_subtask)
        table = super().make_tasks_table(sorted_tasks)
        table.box = rich.box.HORIZONTALS
        if len(table.rows) > subtask_count:
            table.rows[-subtask_count - 1].end_section = True
        return table
