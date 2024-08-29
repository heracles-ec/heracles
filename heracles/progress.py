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
Module for the progress reporting protocol.
"""

from typing import Protocol


class Progress(Protocol):
    """
    Protocol for progress reporting.  Implementations of this protocol are
    meant to be used as a context manager::

        with MyProgress("working") as progress:
            for i in range(100):
                # main progress update
                progress.update(i + 1, 100)

                # report progress of an individual task
                with progress.task(f"subtask {i + 1}") as task:
                    for j in range(10):
                        task.update(j + 1, 10)

    """

    def update(self, current: int | None = None, total: int | None = None) -> None:
        """
        Update progress.
        """

    def task(self, label: str) -> "Progress":
        """
        Create a task with the given label.
        """

    def __enter__(self) -> "Progress":
        """
        Start progress.
        """

    def __exit__(self, *exc) -> None:
        """
        Stop progress.
        """


class NoProgress:
    """
    Dummy progress reporter.
    """

    def update(self, current: int | None = None, total: int | None = None) -> None:
        """
        Dummy progress update (does nothing).
        """
        pass

    def task(self, label: str) -> "NoProgress":
        """
        Create a dummy task (does nothing).
        """
        return NoProgress()

    def __enter__(self) -> "NoProgress":
        """
        Start dummy progress (does nothing).
        """
        return self

    def __exit__(self, *exc) -> None:
        """
        Stop dummy progress (does nothing).
        """
        pass
