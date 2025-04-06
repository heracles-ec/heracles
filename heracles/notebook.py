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

