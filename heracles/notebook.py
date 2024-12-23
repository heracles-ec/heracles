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
from typing import List

class Progress:
    """
    Progress bar using ipywidgets.
    """

    def __init__(self, label: str, *, box: widgets.Box | None = None) -> None:
        if box is None:
            self.box = widgets.VBox()
        else:
            self.box = box
        self.widget = widgets.IntProgress(
            value=0,
            min=0,
            max=1,
            description=label,
            orientation="horizontal",
        )

    def __enter__(self) -> "Progress":
        if not self.box.children:
            display(self.box)
        self.box.children += (self.widget,)
        return self

    def __exit__(self, *exc) -> None:
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

    def update(self, current: int | None = None, total: int | None = None) -> None:
        if current is not None:
            self.widget.value = current
        if total is not None:
            self.widget.max = total

    def task(self, label: str) -> "Progress":
        return self.__class__(label, box=self.box)

class ProgressLogging:
    """
    Progress bar without GUI interface.
    """

    def __init__(self, label: str, *, box: List["ProgressLogging"] = None) -> None:
        self.label = label
        self.current = 0
        self.total = 1  # Default to 1 to avoid division by zero
        self.box = box if box is not None else []
        self.line_offset = len(self.box)  # Track which line to overwrite
        sys.stdout.write("\n")

    def __enter__(self) -> "ProgressLogging":
        # Add this instance to the box if it's not already there
        if self not in self.box:
            self.box.append(self)
        self._display_box()
        #Ensure the cursor ends at a new line after the progress bars
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._display_box()

    def update(self, current: int | None = None, total: int | None = None) -> None:
        # Update progress values
        if current is not None:
            self.current = current
        if total is not None:
            self.total = total
        # Refresh the entire display box
        self._display_box()

    def task(self, label: str) -> "ProgressLogging":
        # Create a new task tied to the same box
        return self.__class__(label, box=self.box)

    def _display_box(self) -> None:
        """
        Redraw the progress bars in the terminal for all tasks in the box.
        """
        # Move the cursor up to overwrite only the progress bar lines
        sys.stdout.write(f"\033[{len(self.box)}F")  # Move up N lines
        sys.stdout.flush()
        # Display all progress bars
        for task in self.box:
            percentage = (task.current / task.total) * 100
            bar_length = 40  # Fixed number of blocks in the bar
            progress_blocks = int(percentage // (100 / bar_length))
            bar = "=" * progress_blocks + " " * (bar_length - progress_blocks)
            sys.stdout.write(f"\r{task.label}: [{bar}]\n")
        sys.stdout.flush()