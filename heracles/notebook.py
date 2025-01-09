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
Module for Jupyter (IPython) notebook integration.
"""

from __future__ import annotations

from .core import external_dependency_explainer

with external_dependency_explainer:
    import ipywidgets as widgets
    from IPython.display import display

import sys
from typing import List, Union

def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        if "IPKernelApp" in get_ipython().config:
            return True
    except Exception:
        return False
    return False

class Progress:
    """
    Combined progress bar that can use ipywidgets or a text-based progress bar.
    """

    def __init__(self, label: str, *, use_widgets: bool = True, box: Union[widgets.Box, List["Progress"]] = None) -> None:
        self.use_widgets = use_widgets
        self.label = label
        self.current = 0
        self.total = 1  # Default to 1 to avoid division by zero

        if self.use_widgets:
            self.box = box if box is not None else widgets.VBox()
            self.widget = widgets.IntProgress(
                value=0,
                min=0,
                max=1,
                description=label,
                orientation="horizontal",
            )
        else:
            if is_notebook():
                raise Exception("use_widgets=False - Cannot use Progress without widgets in notebook.")
            self.box = box if box is not None else []
            self.line_offset = len(self.box)  # Track which line to overwrite
            sys.stdout.write("\n")

    def __enter__(self) -> "Progress":
        if self.use_widgets:
            if not self.box.children:
                display(self.box)
            self.box.children += (self.widget,)
        else:
            if self not in self.box:
                self.box.append(self)
            self._display_terminal()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.use_widgets:
            self.widget.close()
            try:
                index = self.box.children.index(self.widget)
            except ValueError:
                pass
            else:
                self.box.children = (
                    self.box.children[:index] + self.box.children[index + 1 :]
                )
            if not self.box.children:
                self.box.close()
        else:
            self._display_terminal()

    def update(self, current: int | None = None, total: int | None = None) -> None:
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total

        if self.use_widgets:
            self.widget.value = self.current
            self.widget.max = self.total
        else:
            self._display_terminal()

    def task(self, label: str) -> "Progress":
        return self.__class__(label, use_widgets=self.use_widgets, box=self.box)

    def _display_terminal(self) -> None:
        """
        Redraw the progress bars in the terminal for all tasks in the box.
        """
        sys.stdout.write(f"\033[{len(self.box)}F")  # Move curser up N lines to redraw
        sys.stdout.flush()
        for task in self.box:
            percentage = (task.current / task.total) * 100
            bar_length = 40
            progress_blocks = int(percentage // (100 / bar_length))
            bar = "=" * progress_blocks + " " * (bar_length - progress_blocks)
            sys.stdout.write(f"\r{task.label}: [{bar}]\n")
        sys.stdout.flush()