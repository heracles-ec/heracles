'''
Progress (:mod:`heracles.progress`)
===================================

module for progress reporting with rich


.. currentmodule:: heracles.progress

.. autoclass:: ProgressTask
.. autoclass:: Progress
   :exclude-members: add_task, advance, get_default_columns, get_renderable, make_tasks_table, open,
       refresh, remove_task, reset, start, start_task, stop, stop_task, track, update, wrap_file, get_renderables


'''


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
